# here put the import lib
import os
import time
import hashlib
import torch
import numpy as np
from tqdm import tqdm
from generators.data import FullEvalDataset
from trainers.trainer import Trainer
from utils.utils import metric_report, metric_len_report, record_csv, metric_pop_report
from utils.utils import metric_len_5group, metric_pop_5group


class SeqTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)

    def _evaluate_checkpoint(self, epoch):
        # Select the backbone without TSRC, including when main.py is invoked
        # directly. Restore TSRC for validation calibration and the final test.
        model = self._model_ref()
        semantic_enabled = model.use_semantic_rerank
        try:
            model.use_semantic_rerank = False
            return self.eval(epoch=epoch, checkpoint_selection=True)
        finally:
            model.use_semantic_rerank = semantic_enabled
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            inputs = self._prepare_train_inputs(batch)
            loss = self.model(**inputs)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)


    def _model_ref(self):
        return self.model.module if hasattr(self.model, 'module') else self.model


    def _checkpoint_path(self):
        source = getattr(self.args, "source_checkpoint", "")
        if source:
            source = os.path.abspath(source)
            if os.path.isdir(source):
                source = os.path.join(source, "pytorch_model.bin")
            return source
        return os.path.join(self.args.output_dir, "pytorch_model.bin")


    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def _load_eval_checkpoint(self):
        checkpoint_path = self._checkpoint_path()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"semantic_rerank_weight"}
        meaningful_missing = [
            key for key in incompatible.missing_keys
            if key not in allowed_missing
        ]
        if meaningful_missing or incompatible.unexpected_keys:
            self.logger.warning(
                "Checkpoint loaded non-strictly (missing=%s, unexpected=%s)",
                meaningful_missing,
                incompatible.unexpected_keys,
            )

        model = self._model_ref()
        if (
            getattr(self.args, "disable_auto_semantic_weight", False)
            and hasattr(model, "set_semantic_rerank_weight")
        ):
            model.set_semantic_rerank_weight(
                self.args.semantic_rerank_weight
            )

        self.model.to(self.device)
        self.loaded_checkpoint_path = os.path.abspath(checkpoint_path)
        self.loaded_checkpoint_sha256 = self._file_sha256(checkpoint_path)
        self.loaded_checkpoint_epoch = (
            int(checkpoint.get("epoch", -1))
            if isinstance(checkpoint, dict)
            else -1
        )
        self.logger.info(
            "Loaded source checkpoint: %s (sha256=%s)",
            self.loaded_checkpoint_path,
            self.loaded_checkpoint_sha256,
        )
        return self.loaded_checkpoint_epoch


    @staticmethod
    def _target_ranks(scores):
        """Zero-based rank of candidate column 0 (larger score is better)."""
        return scores.gt(scores[:, :1]).sum(dim=1)


    @staticmethod
    def _ndcg_at_10(ranks):
        hits = ranks.lt(10)
        return torch.where(
            hits,
            1.0 / torch.log2(ranks.float() + 2.0),
            torch.zeros_like(ranks, dtype=torch.float32),
        ).mean().item()


    @staticmethod
    def _parse_weight_grid(text):
        values = []
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            value = float(token)
            if value < 0:
                raise ValueError("Semantic residual weights must be non-negative")
            values.append(value)
        return sorted(set(values))


    def _semantic_weight_candidates(self):
        explicit_grid = getattr(self.args, "semantic_weight_grid", "")
        if explicit_grid.strip():
            values = self._parse_weight_grid(explicit_grid)
        elif self._model_ref().use_candidate_standardization:
            step = max(float(self.args.semantic_weight_step), 1e-6)
            max_weight = max(float(self.args.semantic_weight_max), 0.0)
            values = np.arange(
                0.0, max_weight + step / 2.0, step
            ).tolist()
        else:
            values = self._parse_weight_grid(
                self.args.semantic_raw_weight_grid
            )
        if not values:
            raise ValueError("The semantic weight grid cannot be empty")
        return torch.as_tensor(
            values, dtype=torch.float32, device=self.device
        )


    @staticmethod
    def _masked_standardize_candidate_scores(scores, eligible):
        """Standardize per user over eligible unseen candidates only."""
        valid = eligible.to(dtype=scores.dtype)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (scores * valid).sum(dim=1, keepdim=True) / count
        centered = scores - mean
        variance = (centered.square() * valid).sum(dim=1, keepdim=True) / count
        standardized = centered / variance.sqrt().clamp_min(1e-6)
        return standardized.masked_fill(~eligible, 0.0)


    def _prepare_fusion_scores(self, base_scores, semantic_scores, eligible=None):
        model = self._model_ref()
        if not model.use_candidate_standardization:
            return base_scores, semantic_scores
        if eligible is None:
            return (
                model._standardize_candidate_scores(base_scores),
                model._standardize_candidate_scores(semantic_scores),
            )
        return (
            self._masked_standardize_candidate_scores(base_scores, eligible),
            self._masked_standardize_candidate_scores(
                semantic_scores, eligible
            ),
        )


    def _calibrate_semantic_weight(self, base_scores, semantic_scores):
        base_scores, semantic_scores = self._prepare_fusion_scores(
            base_scores, semantic_scores
        )
        best_weight = 0.0
        best_ndcg = -1.0
        best_ranks = None
        for weight in self._semantic_weight_candidates():
            ranks = self._target_ranks(
                base_scores + weight * semantic_scores
            )
            ndcg = self._ndcg_at_10(ranks)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_weight = float(weight.item())
                best_ranks = ranks

        self._model_ref().set_semantic_rerank_weight(best_weight)
        self.logger.info(
            "Calibrated semantic residual weight: %.6g "
            "(validation NDCG@10 %.5f)",
            best_weight,
            best_ndcg,
        )
        return best_ranks


    @staticmethod
    def _full_catalog_ranks(scores, target_indices, eligible):
        scores = scores.masked_fill(~eligible, float("-inf"))
        target_scores = scores.gather(1, target_indices.unsqueeze(1))
        return scores.gt(target_scores).sum(dim=1)


    def _evaluate_full_catalog(self, data_loader, test, calibrate_semantic):
        model = self._model_ref()
        use_semantic = getattr(model, "use_semantic_rerank", False)
        all_items = torch.arange(
            1, self.item_num + 1, dtype=torch.long, device=self.device
        )
        seen_by_user = (
            self.generator.full_test_seen_items
            if test
            else self.generator.full_valid_seen_items
        )

        weights = (
            self._semantic_weight_candidates()
            if calibrate_semantic
            else None
        )
        ranks_by_weight = (
            [[] for _ in range(len(weights))]
            if calibrate_semantic
            else None
        )
        pred_rank_chunks = []
        seq_len_chunks = []
        target_item_chunks = []
        user_id_chunks = []

        for batch in tqdm(
            data_loader, desc="Testing" if test else "Evaluating"
        ):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch, dataset=data_loader.dataset)
            seq = inputs["seq"].long()
            positions = inputs["positions"].long()
            targets = inputs["pos"].long()
            user_ids = inputs["user_id"].long()
            target_indices = targets - 1
            if torch.any(target_indices < 0) or torch.any(
                target_indices >= self.item_num
            ):
                raise ValueError(
                    "Full-catalog target item id is outside the catalog"
                )

            eligible = torch.ones(
                (seq.size(0), self.item_num),
                dtype=torch.bool,
                device=self.device,
            )
            for row, user_id in enumerate(user_ids.detach().cpu().tolist()):
                seen = [
                    item for item in seen_by_user[int(user_id)]
                    if 1 <= item <= self.item_num
                ]
                if seen:
                    seen_indices = torch.as_tensor(
                        seen, dtype=torch.long, device=self.device
                    ) - 1
                    eligible[row, seen_indices] = False
            eligible.scatter_(1, target_indices.unsqueeze(1), True)

            seq_len_chunks.append(seq.gt(0).sum(dim=1).detach().cpu())
            target_item_chunks.append(targets.detach().cpu())
            user_id_chunks.append(user_ids.detach().cpu())

            with torch.no_grad():
                if use_semantic:
                    base_scores, semantic_scores = model.predict(
                        seq=seq,
                        positions=positions,
                        item_indices=all_items,
                        return_semantic_components=True,
                    )
                    base_scores, semantic_scores = self._prepare_fusion_scores(
                        base_scores, semantic_scores, eligible=eligible
                    )
                    if calibrate_semantic:
                        for index, weight in enumerate(weights):
                            ranks = self._full_catalog_ranks(
                                base_scores + weight * semantic_scores,
                                target_indices,
                                eligible,
                            )
                            ranks_by_weight[index].append(
                                ranks.detach().cpu()
                            )
                    else:
                        ranks = self._full_catalog_ranks(
                            base_scores
                            + model.semantic_rerank_weight * semantic_scores,
                            target_indices,
                            eligible,
                        )
                        pred_rank_chunks.append(ranks.detach().cpu())
                else:
                    scores = model.predict(
                        seq=seq,
                        positions=positions,
                        item_indices=all_items,
                    )
                    ranks = self._full_catalog_ranks(
                        scores, target_indices, eligible
                    )
                    pred_rank_chunks.append(ranks.detach().cpu())

        if calibrate_semantic:
            best_weight = 0.0
            best_ndcg = -1.0
            best_ranks = None
            for index, weight in enumerate(weights):
                ranks = torch.cat(ranks_by_weight[index], dim=0)
                ndcg = self._ndcg_at_10(ranks)
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_weight = float(weight.item())
                    best_ranks = ranks
            model.set_semantic_rerank_weight(best_weight)
            self.logger.info(
                "Calibrated semantic residual weight: %.6g "
                "(full-catalog validation NDCG@10 %.5f)",
                best_weight,
                best_ndcg,
            )
            pred_rank = best_ranks
        else:
            pred_rank = torch.cat(pred_rank_chunks, dim=0)

        return (
            pred_rank,
            torch.cat(seq_len_chunks, dim=0),
            torch.cat(target_item_chunks, dim=0),
            torch.cat(user_id_chunks, dim=0),
        )


    def _save_rank_output(
        self, pred_rank, seq_len, target_items, user_ids
    ):
        output_path = getattr(self.args, "rank_output", "")
        if not output_path:
            return
        output_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        model = self._model_ref()
        np.savez_compressed(
            output_path,
            rank=pred_rank.detach().cpu().numpy().astype(np.int64),
            user_id=user_ids.detach().cpu().numpy().astype(np.int64),
            target_item=target_items.detach().cpu().numpy().astype(np.int64),
            sequence_length=seq_len.detach().cpu().numpy().astype(np.int64),
            semantic_weight=np.asarray(
                float(model.semantic_rerank_weight.item())
                if hasattr(model, "semantic_rerank_weight")
                else 0.0,
                dtype=np.float64,
            ),
            source_checkpoint=np.asarray(
                getattr(self, "loaded_checkpoint_path", "")
            ),
            source_checkpoint_sha256=np.asarray(
                getattr(self, "loaded_checkpoint_sha256", "")
            ),
        )
        self.logger.info("Per-user ranks saved to %s", output_path)



    def eval(self, epoch=0, test=False, load_checkpoint=True, checkpoint_selection=False):

        print('')
        model = self._model_ref()
        if test:
            self.logger.info(
                "\n----------------------------------------------------------------"
            )
            self.logger.info("********** Running test **********")
            desc = "Testing"
            if load_checkpoint:
                self._load_eval_checkpoint()
                model = self._model_ref()
                # Every test-only ablation first chooses its scalar weight on
                # validation using this exact source checkpoint.  Test labels
                # are never involved in calibration.
                if (
                    getattr(model, "use_semantic_rerank", False)
                    and not self.args.disable_auto_semantic_weight
                ):
                    self.eval(
                        epoch=self.loaded_checkpoint_epoch,
                        test=False,
                        load_checkpoint=False,
                    )
            data_loader = self.test_loader
            if getattr(model, "use_semantic_rerank", False):
                self.logger.info(
                    "Using semantic residual weight: %.6g",
                    float(model.semantic_rerank_weight.item()),
                )
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info(
                "********** Epoch: %d eval **********" % epoch
            )
            desc = "Evaluating"
            data_loader = (
                self.checkpoint_valid_loader
                if checkpoint_selection
                else self.valid_loader
            )

        self.model.eval()
        calibrate_semantic = (
            not test
            and not self.args.disable_auto_semantic_weight
            and getattr(model, "use_semantic_rerank", False)
        )

        if isinstance(data_loader.dataset, FullEvalDataset):
            pred_rank, seq_len, target_items, user_ids = (
                self._evaluate_full_catalog(
                    data_loader, test, calibrate_semantic
                )
            )
        else:
            pred_rank_chunks = []
            seq_len_chunks = []
            target_item_chunks = []
            base_score_chunks = []
            semantic_score_chunks = []

            for batch in tqdm(data_loader, desc=desc):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = self._prepare_eval_inputs(batch, dataset=data_loader.dataset)
                seq_len_chunks.append(
                    torch.sum(inputs["seq"] > 0, dim=1).detach().cpu()
                )
                target_item_chunks.append(inputs["pos"].detach().cpu())

                with torch.no_grad():
                    inputs["item_indices"] = torch.cat(
                        [inputs["pos"].unsqueeze(1), inputs["neg"]],
                        dim=1,
                    )
                    if calibrate_semantic:
                        base_scores, semantic_scores = self.model.predict(
                            **inputs, return_semantic_components=True
                        )
                        base_score_chunks.append(base_scores)
                        semantic_score_chunks.append(semantic_scores)
                    else:
                        scores = self.model.predict(**inputs)
                        pred_rank_chunks.append(
                            self._target_ranks(scores).detach().cpu()
                        )

            if calibrate_semantic:
                pred_rank = self._calibrate_semantic_weight(
                    torch.cat(base_score_chunks, dim=0),
                    torch.cat(semantic_score_chunks, dim=0),
                ).detach().cpu()
            else:
                pred_rank = torch.cat(pred_rank_chunks, dim=0)
            seq_len = torch.cat(seq_len_chunks, dim=0)
            target_items = torch.cat(target_item_chunks, dim=0)
            split_user_ids = (
                self.generator.test_user_ids
                if test
                else self.generator.valid_user_ids
            )
            user_ids = torch.as_tensor(split_user_ids, dtype=torch.long)

        self.logger.info('')
        rank_array = pred_rank.detach().cpu().numpy()
        seq_len_array = seq_len.detach().cpu().numpy()
        target_item_array = target_items.detach().cpu().numpy()
        res_dict = metric_report(rank_array)
        res_len_dict = metric_len_report(
            rank_array,
            seq_len_array,
            aug_len=self.args.aug_seq_len,
            args=self.args,
        )
        res_pop_dict = metric_pop_report(
            rank_array,
            self.item_pop,
            target_item_array,
            args=self.args,
        )

        self.logger.info("Overall Performance:")
        for key, value in res_dict.items():
            if not test:
                self.writer.add_scalar(
                    "Validation/{}".format(key), value, epoch
                )
            self.logger.info('\t %s: %.5f' % (key, value))

        if test:
            self.logger.info("User Group Performance:")
            for key, value in res_len_dict.items():
                self.logger.info('\t %s: %.5f' % (key, value))
            self.logger.info("Item Group Performance:")
            for key, value in res_pop_dict.items():
                self.logger.info('\t %s: %.5f' % (key, value))

        res_dict = {**res_dict, **res_len_dict, **res_pop_dict}

        if test:
            record_csv(self.args, res_dict)
            self._save_rank_output(
                pred_rank, seq_len, target_items, user_ids
            )

        return res_dict
    

    def test(self):
        """Evaluate a checkpoint and report the epoch stored in it."""
        res = self.eval(test=True)
        return res, getattr(self, "loaded_checkpoint_epoch", -1)



    def save_user_emb(self):

        self._load_eval_checkpoint()
        test_loader = self.test_loader

        self.model.eval()
        user_emb = torch.empty(0).to(self.device)
        desc = 'Running'

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            
            with torch.no_grad():

                per_user_emb = self.model.get_user_emb(**inputs)
                user_emb = torch.cat([user_emb, per_user_emb], dim=0)
        
        user_emb = user_emb.detach().cpu().numpy()
        import pickle
        pickle.dump(user_emb, open("./usr_emb_sasrec.pkl", "wb"))


    
    def test_group(self):

        if self.args.eval_mode == "full":
            # eval(test=True) already reports sequence-length and popularity
            # groups while respecting full-catalog history masks.
            return self.eval(test=True)

        print('')
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running Group test **********")
        desc = 'Testing'
        self._load_eval_checkpoint()
        model = self._model_ref()
        if (
            getattr(model, "use_semantic_rerank", False)
            and not self.args.disable_auto_semantic_weight
        ):
            self.eval(
                epoch=self.loaded_checkpoint_epoch,
                test=False,
                load_checkpoint=False,
            )
        test_loader = self.test_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        # res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        # res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)
        hr_len, ndcg_len, count_len = metric_len_5group(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), [5, 10, 15, 20])
        hr_pop, ndcg_pop, count_pop = metric_pop_5group(pred_rank.detach().cpu().numpy(), self.item_pop,  target_items.detach().cpu().numpy(), [10, 30, 60, 100])

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            self.logger.info('\t %s: %.5f' % (k, v))

        self.logger.info("User Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_len, ndcg_len)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        self.logger.info("Item Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_pop, ndcg_pop)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        
        
        return res_dict
    


class CL4SRecTrainer(SeqTrainer):

    def __init__(self, args, logger, writer, device, generator):
        
        super().__init__(args, logger, writer, device, generator)


    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            seq, pos, neg, positions, aug1, aug2 = batch
            seq, pos, neg, positions, aug1, aug2 = seq.long(), pos.long(), neg.long(), positions.long(), aug1.long(), aug2.long()
            aug = (aug1, aug2)
            loss = self.model(seq, pos, neg, positions, aug)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



class SSEPTTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            seq_user, pos_user, neg_user, seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            seq_user, pos_user, neg_user = seq_user.long(), pos_user.long(), neg_user.long()
            loss = self.model(seq_user, pos_user, neg_user, seq, pos, neg, positions)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            try:
                self.model.load_state_dict(model_state_dict['state_dict'])
            except:
                self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            seq_user, pos_user, neg_user, seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            seq_user, pos_user, neg_user = seq_user.long(), pos_user.long(), neg_user.long()
            seq_len = torch.cat([seq_len, torch.sum(seq>0, dim=1)])

            with torch.no_grad():

                pred_logits = -self.model.predict(seq_user, seq, torch.cat([pos_user.unsqueeze(1), neg_user], dim=1), torch.cat([pos.unsqueeze(1), neg], dim=1), positions)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len)
        
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        for k, v in res_len_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict}

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict
