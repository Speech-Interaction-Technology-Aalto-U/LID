"""This script is used to convert the Voice Privacy Challenge 2026 (VPC26) .pt files to parquet format for the LID repository.
We assume that the .pt files are located in the following paths from the root Voice Privacy Challenge 2026 (VPC26) repository:
- exp/asv_anon{anon_data_suffix}/cosine_out/emb_xvect/libri_dev_enrolls{anon_data_suffix}/utt-level/speaker_vectors.pt
- exp/asv_anon{anon_data_suffix}/cosine_out/emb_xvect/libri_dev_trials_mixed{anon_data_suffix}/utt-level/speaker_vectors.pt
- exp/asv_anon{anon_data_suffix}/cosine_out/emb_xvect/libri_test_enrolls{anon_data_suffix}/utt-level/speaker_vectors.pt
- exp/asv_anon{anon_data_suffix}/cosine_out/emb_xvect/libri_test_trials_mixed{anon_data_suffix}/utt-level/speaker_vectors.pt

The id2idx mapping files are located in the following paths:
- exp/asv_anon{anon_data_suffix}/cosine_out/emb_xvect/libri_dev_enrolls{anon_data_suffix}/utt-level/id2idx
- exp/asv_anon{anon_data_suffix}/cosine_out/emb_xvect/libri_dev_trials_mixed{anon_data_suffix}/utt-level/id2idx
- exp/asv_anon{anon_data_suffix}/cosine_out/emb_xvect/libri_test_enrolls{anon_data_suffix}/utt-level/id2idx
- exp/asv_anon{anon_data_suffix}/cosine_out/emb_xvect/libri_test_trials_mixed{anon_data_suffix}/utt-level/id2idx

The output parquet file will be saved in the current working directory as "embeddings.parquet".
"""
import torch
import pandas as pd
import argparse


def convert_VPC26_pt_files_to_parquet(
        anon_data_suffix: str = "_B5",
        is_semi_informed: bool = True
    ):
    """
    This function opens the .pt files containing the embeddings and the id2idx mapping files, merges them into a single parquet file.
    It has been made to convert the Voice Privacy Challenge 2026 (VPC26) .pt files to parquet format for the LID repositery.

    Args:
        anon_data_suffix (str): The suffix of the anonymized data. Default is "_B5".
        is_semi_informed (bool): Whether the data is semi-informed or not. Default is True.
    """

    if is_semi_informed:
        base_path = f"exp/asv_anon{anon_data_suffix}/cosine_out/emb_xvect/"
    else:
        anon_data_suffix = ""
        base_path = "exp/asv_ssl/cosine_out/emb_xvect"

    # define the .pt files paths
    file_dev_enrolls = f"{base_path}/libri_dev_enrolls{anon_data_suffix}/utt-level/speaker_vectors.pt"
    file_dev_trials = f"{base_path}/libri_dev_trials_mixed{anon_data_suffix}/utt-level/speaker_vectors.pt"
    file_test_enrolls = f"{base_path}/libri_test_enrolls{anon_data_suffix}/utt-level/speaker_vectors.pt"
    file_test_trials = f"{base_path}/libri_test_trials_mixed{anon_data_suffix}/utt-level/speaker_vectors.pt"

    # define the id2idx files paths
    id2idx_dev_enrolls = f"{base_path}/libri_dev_enrolls{anon_data_suffix}/utt-level/id2idx"
    id2idx_dev_trials = f"{base_path}/libri_dev_trials_mixed{anon_data_suffix}/utt-level/id2idx"
    id2idx_test_enrolls = f"{base_path}/libri_test_enrolls{anon_data_suffix}/utt-level/id2idx"
    id2idx_test_trials = f"{base_path}/libri_test_trials_mixed{anon_data_suffix}/utt-level/id2idx"

    merged_data = []

    for file, id2idx_file in [(file_dev_enrolls, id2idx_dev_enrolls), (file_dev_trials, id2idx_dev_trials), (file_test_enrolls, id2idx_test_enrolls), (file_test_trials, id2idx_test_trials)]:
        # load the .pt file
        try:
            data = torch.load(file)
            print(f"Loaded data from {file}: {data.shape}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file}. Please ensure this python file is in the root of the VPC26 repository and that the path {file} is correct.")

        # open the id2idx file and read the mapping into a dictionary
        try:
            with open(id2idx_file, "r") as f:
                id2idx_dict = {}
                for line in f:
                    key, value = line.strip().split()
                    id2idx_dict[key] = int(value)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {id2idx_file}. Please ensure this python file is in the root of the VPC26 repository and that the path {id2idx_file} is correct.")

        # Merge the data and the id2idx mapping into a single parquet file
        # it must contains utterance_id, speaker_id, and embedding columns
        # the speaker_id is the first part of the utterance_id, i.e. 1272-128104-0000 -> 1272
        # the id2idx mapping is used to get the index of the embedding in the data tensor
        for utt_id, idx in id2idx_dict.items():
            # get the speaker_id from the utterance_id
            speaker_id = utt_id.split('-')[0]

            # get the embedding from the data tensor using the index
            embedding = data[idx].cpu().numpy()

            merged_data.append({
                "utterance_id": utt_id,
                "speaker_id": speaker_id,
                "embedding": embedding
            })

    # save the merged data to a parquet file
    output_file = "embeddings.parquet"
    df = pd.DataFrame(merged_data)
    df.to_parquet(output_file, engine='pyarrow', index=False)
    print(f"Merged data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VPC26 .pt files to parquet format.")
    parser.add_argument("--anon_data_suffix", type=str, default="_B5", help="Suffix of the anonymized data.")
    parser.add_argument("--is_semi_informed", type=bool, default=True, help="Whether the data has been computed using a semi-informed approach or not.")
    args = parser.parse_args()
    convert_VPC26_pt_files_to_parquet(
        anon_data_suffix=args.anon_data_suffix,
        is_semi_informed=args.is_semi_informed
    )
    # e.g. of command line usage:
    # python open_pt_and_merge_to_parquet.py --anon_data_suffix "_B5" --is_semi_informed True