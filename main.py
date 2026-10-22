# here put the import lib
import os
import argparse
import json
import time
import torch

from generators.generator import Seq2SeqGeneratorAllUser
from generators.generator import GeneratorAllUser
from generators.bert_generator import BertGeneratorAllUser
from trainers.sequence_trainer import SeqTrainer
from utils.utils import set_seed
from utils.logger import Logger


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name", 
                    default='llmesr_sasrec',
                    choices=[
                    "llmesr_sasrec", "llmesr_bert4rec", "llmesr_gru4rec",
                    ],
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset", 
                    default="yelp", 
                    choices=["yelp", "fashion", "beauty",],  # preprocess by myself
                    help="Choose the dataset")
parser.add_argument("--inter_file",
                    default="inter",
                    type=str,
                    help="the name of interaction file")
parser.add_argument("--demo", 
                    default=False, 
                    action='store_true', 
                    help='whether run demo')
parser.add_argument("--pretrain_dir",
                    type=str,
                    default="sasrec_seq",
                    help="the path that pretrained model saved in")
parser.add_argument("--output_dir",
                    default='./saved/',
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--check_path",
                    default='',
                    type=str,
                    help="the save path of checkpoints for different running")
parser.add_argument("--source_checkpoint",
                    default='',
                    type=str,
                    help="checkpoint file/directory shared by test-only ablations")
parser.add_argument("--experiment_variant",
                    default='',
                    type=str,
                    help="explicit ablation name stored in CSV/JSON outputs")
parser.add_argument("--result_json",
                    default='',
                    type=str,
                    help="optional structured result JSON path")
parser.add_argument("--rank_output",
                    default='',
                    type=str,
                    help="optional .npz path for paired per-user test ranks")
parser.add_argument("--do_test",
                    default=False,
                    action="store_true",
                    help="whehther run the test on the well-trained model")
parser.add_argument("--do_emb",
                    default=False,
                    action="store_true",
                    help="save the user embedding derived from the SRS model")
parser.add_argument("--do_group",
                    default=False,
                    action="store_true",
                    help="conduct the group test")
parser.add_argument("--keepon",
                    default=False,
                    action="store_true",
                    help="whether keep on training based on a trained model")
parser.add_argument("--keepon_path",
                    type=str,
                    default="normal",
                    help="the path of trained model for keep on training")
parser.add_argument("--clip_path",
                    type=str,
                    default="",
                    help="the path to save the CLIP-pretrained embedding and adapter")
parser.add_argument("--ts_user",
                    type=int,
                    default=None,
                    help="short-history threshold (dataset default: Yelp 12, Fashion 3, Beauty 9)")
parser.add_argument("--ts_item",
                    type=int,
                    default=None,
                    help="tail-item threshold (dataset default: Yelp 13, Fashion 4, Beauty 4)")

# Model parameters
parser.add_argument("--hidden_size",
                    default=64,
                    type=int,
                    help="the hidden size of embedding")
parser.add_argument("--trm_num",
                    default=2,
                    type=int,
                    help="the number of transformer layer")
parser.add_argument("--num_heads",
                    default=1,
                    type=int,
                    help="the number of heads in Trm layer")
parser.add_argument("--num_layers",
                    default=1,
                    type=int,
                    help="the number of GRU layers")
parser.add_argument("--cl_scale",
                    type=float,
                    default=0.1,
                    help="the scale for contastive loss")
parser.add_argument("--mask_crop_ratio",
                    type=float,
                    default=0.3,
                    help="the mask/crop ratio for CL4SRec")
parser.add_argument("--tau",
                    default=1,
                    type=float,
                    help="the temperature for contrastive loss")
parser.add_argument("--sse_ratio",
                    default=0.4,
                    type=float,
                    help="the sse ratio for SSE-PT model")
parser.add_argument("--dropout_rate",
                    default=0.5,
                    type=float,
                    help="the dropout rate")
parser.add_argument("--max_len",
                    default=200,
                    type=int,
                    help="the max length of input sequence")
parser.add_argument("--mask_prob",
                    type=float,
                    default=0.6,
                    help="the mask probability for training Bert model")
parser.add_argument("--aug",
                    default=False,
                    action="store_true",
                    help="whether augment the sequence data")
parser.add_argument("--aug_seq",
                    default=False,
                    action="store_true",
                    help="whether use the augmented data")
parser.add_argument("--aug_seq_len",
                    default=0,
                    type=int,
                    help="the augmented length for each sequence")
parser.add_argument("--aug_file",
                    default="inter",
                    type=str,
                    help="the augmentation file name")
parser.add_argument("--train_neg",
                    default=1,
                    type=int,
                    help="the number of negative samples for training")
parser.add_argument("--test_neg",
                    default=100,
                    type=int,
                    help="the number of negative samples for test")
parser.add_argument("--checkpoint_eval_mode",
                    default="full",
                    choices=["full", "sampled"],
                    help="validation candidate protocol for selecting the backbone checkpoint (default: full)")
parser.add_argument("--eval_mode",
                    default="sampled",
                    choices=["sampled", "full"],
                    help="candidate protocol for test-time evaluation and semantic calibration (default: sampled)")
parser.add_argument("--eval_batch_size",
                    default=16,
                    type=int,
                    help="user batch size for full-catalog evaluation")
parser.add_argument('--eval_seed',
                    type=int,
                    default=None,
                    help="fixed seed for sampled validation/test candidates (defaults to --seed)")
parser.add_argument("--suffix_num",
                    default=5,
                    type=int,
                    help="the suffix number for augmented sequence")
parser.add_argument("--prompt_num",
                    default=2,
                    type=int,
                    help="the number of prompts")
parser.add_argument("--freeze",
                    default=False,
                    action="store_true",
                    help="whether freeze the pretrained architecture when finetuning")
parser.add_argument("--pg",
                    default="length",
                    choices=['length', 'attention'],
                    type=str,
                    help="choose the prompt generator")
parser.add_argument("--use_cross_att",
                    default=False,
                    action="store_true",
                    help="whether add a cross-attention to interact the dual-view")
parser.add_argument("--disable_semantic_rerank",
                    default=False,
                    action="store_true",
                    help="disable MCRR completely (the true Base condition)")
parser.add_argument("--disable_semantic_centering",
                    default=False,
                    action="store_true",
                    help="C ablation: do not remove the catalog mean direction")
parser.add_argument("--semantic_pooling",
                    default="recency",
                    choices=["recency", "mean"],
                    help="R mechanism: recency-aware or uniform valid-history pooling")
parser.add_argument("--disable_candidate_standardization",
                    default=False,
                    action="store_true",
                    help="Z ablation: fuse raw backbone and semantic scores")
parser.add_argument("--semantic_recency_decay",
                    default=0.2,
                    type=float,
                    help="exponential decay gamma for older valid interactions")
parser.add_argument("--semantic_rerank_weight",
                    default=1.0,
                    type=float,
                    help="fixed residual weight when automatic calibration is off")
parser.add_argument("--disable_auto_semantic_weight",
                    default=False,
                    action="store_true",
                    help="use the fixed semantic_rerank_weight without validation search")
parser.add_argument("--semantic_weight_max",
                    default=3.0,
                    type=float,
                    help="maximum validation weight when Z is enabled")
parser.add_argument("--semantic_weight_step",
                    default=0.1,
                    type=float,
                    help="validation weight step when Z is enabled")
parser.add_argument("--semantic_weight_grid",
                    default='',
                    type=str,
                    help="optional comma-separated grid overriding automatic grids")
parser.add_argument("--semantic_raw_weight_grid",
                    default="0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100",
                    type=str,
                    help="validation grid used when candidate standardization is off")
parser.add_argument("--alpha",
                    default=0.1,
                    type=float,
                    help="the weight of auxiliary loss")
parser.add_argument("--user_sim_func",
                    default="kd",
                    type=str,
                    help="the type of user similarity function to derive the loss")
parser.add_argument("--item_reg",
                    default=False,
                    action="store_true",
                    help="whether regularize the item embedding by CL")
parser.add_argument("--beta",
                    default=0.1,
                    type=float,
                    help="the weight of regulation loss")
parser.add_argument("--sim_user_num",
                    default=10,
                    type=int,
                    help="the number of similar users for enhancement")
parser.add_argument("--split_backbone",
                    default=False,
                    action="store_true",
                    help="whether use a split backbone")
parser.add_argument("--co_view",
                    default=False,
                    action="store_true",
                    help="only use the collaborative view")
parser.add_argument("--se_view",
                    default=False,
                    action="store_true",
                    help="only use the semantic view")


# Other parameters
parser.add_argument("--train_batch_size",
                    default=128,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--lr",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2",
                    default=0,
                    type=float,
                    help='The L2 regularization')
parser.add_argument("--num_train_epochs",
                    default=200,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_dc_step",
                    default=1000,
                    type=int,
                    help='every n step, decrease the lr')
parser.add_argument("--lr_dc",
                    default=0,
                    type=float,
                    help='how many learning rate to decrease')
parser.add_argument("--patience",
                    type=int,
                    default=20,
                    help='How many steps to tolerate the performance decrease while training')
parser.add_argument("--watch_metric",
                    type=str,
                    default='NDCG@10',
                    help="which metric is used to select model.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--gpu_id',
                    default=0,
                    type=int,
                    help='The device id.')
parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='The number of workers in dataloader')
parser.add_argument("--log", 
                    default=False,
                    action="store_true",
                    help="whether create a new log file")
parser.add_argument("--tensorboard",
                    default=False,
                    action="store_true",
                    help="write TensorBoard events; optional and disabled by default")

parser.add_argument("--log_root",
                    type=str,
                    default="log",
                    help="root directory for text, CSV and TensorBoard logs")

torch.autograd.set_detect_anomaly(True)

args = parser.parse_args()
if args.eval_seed is None:
    args.eval_seed = args.seed
dataset_thresholds = {"yelp": (12, 13), "fashion": (3, 4), "beauty": (9, 4)}
default_ts_user, default_ts_item = dataset_thresholds[args.dataset]
if args.ts_user is None:
    args.ts_user = default_ts_user
if args.ts_item is None:
    args.ts_item = default_ts_item
set_seed(args.seed) # fix the random seed
args.output_dir = os.path.join(args.output_dir, args.dataset)
args.pretrain_dir = os.path.join(args.output_dir, args.pretrain_dir)
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.keepon_path = os.path.join(args.output_dir, args.keepon_path)
args.output_dir = os.path.join(args.output_dir, args.check_path)    # if check_path is none, then without check_path


def main():

    run_started = time.perf_counter()
    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger
    args.now_str = log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")


    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    if args.model_name in ['llmesr_gru4rec']:
        generator = GeneratorAllUser(args, logger, device)
    elif args.model_name in ["llmesr_bert4rec"]:
        generator = BertGeneratorAllUser(args, logger, device)
    elif args.model_name in ["llmesr_sasrec"]:
        generator = Seq2SeqGeneratorAllUser(args, logger, device)
    else:
        raise ValueError

    trainer = SeqTrainer(args, logger, writer, device, generator)

    result = None
    best_epoch = -1
    if args.do_test:
        result, best_epoch = trainer.test()
    elif args.do_emb:
        trainer.save_user_emb()
    elif args.do_group:
        result = trainer.test_group()
    else:
        result, best_epoch = trainer.train()

    if args.result_json and result is not None:
        model = (
            trainer.model.module
            if hasattr(trainer.model, "module")
            else trainer.model
        )
        checkpoint_path = getattr(
            trainer,
            "loaded_checkpoint_path",
            os.path.abspath(
                os.path.join(args.output_dir, "pytorch_model.bin")
            ),
        )
        checkpoint_sha256 = getattr(
            trainer, "loaded_checkpoint_sha256", ""
        )
        selected_weight = (
            float(model.semantic_rerank_weight.item())
            if (
                hasattr(model, "semantic_rerank_weight")
                and getattr(model, "use_semantic_rerank", False)
            )
            else None
        )
        payload = {
            "schema_version": 1,
            "completed_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime()
            ),
            "dataset": args.dataset,
            "backbone": args.model_name,
            "seed": int(args.seed),
            "eval_seed": int(args.eval_seed),
            "variant": args.experiment_variant or args.check_path,
            "mcrr": {
                "enabled": not bool(args.disable_semantic_rerank),
                "catalog_centering_C": not bool(
                    args.disable_semantic_centering
                ),
                "recency_pooling_R": args.semantic_pooling == "recency",
                "candidate_standardization_Z": not bool(
                    args.disable_candidate_standardization
                ),
                "pooling": args.semantic_pooling,
                "recency_decay": float(args.semantic_recency_decay),
                "automatic_weight": not bool(
                    args.disable_auto_semantic_weight
                ),
                "selected_weight": selected_weight,
            },
            "metrics": {
                key: float(value) for key, value in result.items()
            },
            "best_epoch": int(best_epoch),
            "runtime_seconds": float(
                time.perf_counter() - run_started
            ),
            "source_checkpoint": checkpoint_path,
            "source_checkpoint_sha256": checkpoint_sha256,
            "protocol": {
                "evaluation": args.eval_mode,
                "checkpoint_selection_evaluation": args.checkpoint_eval_mode,
                "test_negatives": (
                    int(args.test_neg)
                    if args.eval_mode == "sampled"
                    else None
                ),
                "candidates_per_user": (
                    int(args.test_neg) + 1
                    if args.eval_mode == "sampled"
                    else "all_unseen_items"
                ),
                "catalog_items": (
                    int(trainer.item_num)
                    if args.eval_mode == "full"
                    else None
                ),
            },
            "parameters": {
                "hidden_size": int(args.hidden_size),
                "max_len": int(args.max_len),
                "mask_prob": float(args.mask_prob),
                "use_cross_att": bool(args.use_cross_att),
                "freeze": bool(args.freeze),
                "semantic_weight_grid": args.semantic_weight_grid,
                "semantic_raw_weight_grid": args.semantic_raw_weight_grid,
                "semantic_weight_max": float(args.semantic_weight_max),
                "semantic_weight_step": float(args.semantic_weight_step),
            },
        }
        result_path = os.path.abspath(args.result_json)
        result_dir = os.path.dirname(result_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        temporary_result_path = result_path + ".tmp"
        with open(temporary_result_path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(temporary_result_path, result_path)
        logger.info("Structured result saved to %s", result_path)

    log_manager.end_log()   # delete the logger threads



if __name__ == "__main__":

    main()
