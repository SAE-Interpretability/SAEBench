import os
import json
import torch

import sae_bench.custom_saes.custom_sae_config as custom_sae_config
import sae_bench.custom_saes.relu_sae as relu_sae
import sae_bench.custom_saes.run_all_evals_custom_saes as run_all_evals_custom_saes

import sae_bench.evals.core.main as core
import sae_bench.evals.sparse_probing.main as sparse_probing

import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex

import sae_bench.sae_bench_utils.graphing_utils as graphing_utils


################################################################################
# Basic Setup
################################################################################

RANDOM_SEED = 42

output_folders = {
    "absorption": "eval_results/absorption",
    "autointerp": "eval_results/autointerp",
    "core": "eval_results/core",
    "scr": "eval_results/scr",
    "tpp": "eval_results/tpp",
    "sparse_probing": "eval_results/sparse_probing",
    "unlearning": "eval_results/unlearning",
}

eval_types = [
    "absorption",
    # "autointerp",   # autointerp 必须单独运行
    "core",
    "scr",
    "tpp",
    "sparse_probing",
    # "unlearning",
]

if "autointerp" in eval_types:
    raise ValueError("autointerp must be run using a python script")

device = general_utils.setup_environment()

model_name = "pythia-70m-deduped"
llm_batch_size = 2048
torch_dtype = torch.float32
str_dtype = torch_dtype.__str__().split(".")[-1]

save_activations = False


################################################################################
# Load Custom SAE
################################################################################

repo_id = "canrager/lm_sae"
baseline_filename = (
    "pythia70m_sweep_standard_ctx128_0712/resid_post_layer_4/trainer_8/ae.pt"
)

hook_layer = 4
hook_name = f"blocks.{hook_layer}.hook_resid_post"

sae = relu_sae.load_dictionary_learning_relu_sae(
    repo_id,
    baseline_filename,
    model_name,
    device,
    torch_dtype,
    layer=hook_layer,
)

print(f"sae dtype: {sae.dtype}, device: {sae.device}")

d_sae, d_in = sae.W_dec.data.shape
assert d_sae >= d_in
print(f"d_in: {d_in}, d_sae: {d_sae}")


################################################################################
# SAE Config
################################################################################

sae.cfg = custom_sae_config.CustomSAEConfig(
    model_name,
    d_in=d_in,
    d_sae=d_sae,
    hook_name=hook_name,
    hook_layer=hook_layer,
)

sae.cfg.dtype = str_dtype

new_sae_key = "vanilla"

trainer_markers = {
    "standard": "o",
    "jumprelu": "X",
    "topk": "^",
    "p_anneal": "*",
    "gated": "d",
    new_sae_key: "s",
}

trainer_colors = {
    "standard": "blue",
    "jumprelu": "orange",
    "topk": "green",
    "p_anneal": "red",
    "gated": "purple",
    new_sae_key: "black",
}
sae.cfg.context_size = 128
sae.cfg.architecture = new_sae_key
sae.cfg.training_tokens = 200_000_000



################################################################################
# Select SAEs to Eval
################################################################################

unique_custom_sae_id = baseline_filename.replace("/", "_").replace(".", "_")
print(f"sae_id: {unique_custom_sae_id}")

custom_saes = [(unique_custom_sae_id, sae)]

sae_regex_pattern = r"(sae_bench_pythia70m_sweep_topk_ctx128_0730).*"
sae_block_pattern = r".*blocks\.([4])\.hook_resid_post__trainer_(8)$"

baseline_saes = get_saes_from_regex(sae_regex_pattern, sae_block_pattern)
print(f"baseline_saes: {baseline_saes}")

baseline_sae_id = f"{baseline_saes[0][0]}_{baseline_saes[0][1]}".replace(".", "_")
print(f"baseline_sae_id: {baseline_sae_id}")

selected_saes = custom_saes + baseline_saes


################################################################################
# Run Core Eval
################################################################################

_ = core.multiple_evals(
    selected_saes=selected_saes,
    n_eval_reconstruction_batches=200,
    n_eval_sparsity_variance_batches=200,
    eval_batch_size_prompts=512,
    compute_featurewise_density_statistics=True,
    compute_featurewise_weight_based_metrics=True,
    exclude_special_tokens_from_reconstruction=True,
    dataset="Skylion007/openwebtext",
    context_size=128,
    output_folder="eval_results/core",
    verbose=True,
    dtype=str_dtype,
)


################################################################################
# Sparse Probing
################################################################################

dataset_names = ["LabHC/bias_in_bios_class_set1"]

_ = sparse_probing.run_eval(
    sparse_probing.SparseProbingEvalConfig(
        model_name=model_name,
        random_seed=RANDOM_SEED,
        llm_batch_size=llm_batch_size,
        llm_dtype=str_dtype,
        dataset_names=dataset_names,
    ),
    selected_saes,
    device,
    "eval_results/sparse_probing",
    force_rerun=False,
    clean_up_activations=True,
    save_activations=save_activations,
)


################################################################################
# Load Results and Plot
################################################################################

image_path = "./images"
os.makedirs(image_path, exist_ok=True)

results_folders = ["./eval_results"]
eval_type = "sparse_probing"

eval_folders = [f"{folder}/{eval_type}" for folder in results_folders]
core_folders = [f"{folder}/core" for folder in results_folders]

eval_filenames = graphing_utils.find_eval_results_files(eval_folders)
core_filenames = graphing_utils.find_eval_results_files(core_folders)

print(f"eval_filenames: {eval_filenames}")
print(f"core_filenames: {core_filenames}")

eval_results_dict = graphing_utils.get_eval_results(eval_filenames)
core_results_dict = graphing_utils.get_eval_results(core_filenames)

for sae_id in eval_results_dict:
    eval_results_dict[sae_id].update(core_results_dict[sae_id])


print(eval_results_dict.keys())

# Example: Compare metrics
baseline_filepath = eval_filenames[0]
with open(baseline_filepath) as f:
    baseline_sae_eval_results = json.load(f)

custom_filepath = eval_filenames[1]
with open(custom_filepath) as f:
    custom_sae_eval_results = json.load(f)

k = 1

print("Baseline SAE top-1 =", baseline_sae_eval_results["eval_result_metrics"]["sae"][f"sae_top_{k}_test_accuracy"])
print("Custom SAE top-1   =", custom_sae_eval_results["eval_result_metrics"]["sae"][f"sae_top_{k}_test_accuracy"])
print("LLM top-1          =", baseline_sae_eval_results["eval_result_metrics"]["llm"][f"llm_top_{k}_test_accuracy"])


################################################################################
# Plotting
################################################################################

image_base_name = os.path.join(image_path, "sparse_probing")

graphing_utils.plot_results(
    eval_filenames,
    core_filenames,
    eval_type,
    image_base_name,
    k,
    trainer_markers=trainer_markers,
    trainer_colors=trainer_colors,
)


################################################################################
# Run all evals
################################################################################

_ = run_all_evals_custom_saes.run_evals(
    model_name,
    selected_saes,
    llm_batch_size,
    str_dtype,
    device,
    eval_types,
    api_key=None,
    force_rerun=False,
    save_activations=save_activations,
)


################################################################################
# Plot results again
################################################################################

for eval_type in eval_types:
    eval_folders = [f"{folder}/{eval_type}" for folder in results_folders]
    eval_filenames = graphing_utils.find_eval_results_files(eval_folders)

    graphing_utils.plot_results(
        eval_filenames,
        core_filenames,
        eval_type,
        image_base_name,
        k=10,
        trainer_markers=trainer_markers,
        trainer_colors=trainer_colors,
    )