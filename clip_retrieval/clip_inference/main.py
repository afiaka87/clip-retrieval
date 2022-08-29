"""main module combines distributor, runner, reader, mapper, writer to produce clip embeddings"""

from glob import glob
from pathlib import Path
from braceexpand import braceexpand
import fire
from clip_retrieval.clip_inference.load_clip import load_clip, vae_preprocess, clip_transform
from clip_retrieval.clip_inference.logger import LoggerReader, LoggerWriter
from clip_retrieval.clip_inference.reader import folder_to_keys

from clip_retrieval.clip_inference.mapper import ClipMapper
from clip_retrieval.clip_inference.reader import FilesReader, WebdatasetReader
from clip_retrieval.clip_inference.writer import NumpyWriter
from clip_retrieval.clip_inference.distributor import (
    PysparkDistributor,
    SequentialDistributor,
)
from clip_retrieval.clip_inference.runner import Runner


def main(
    input_dataset,
    output_folder,
    input_format="files",
    cache_path=None,
    batch_size=256,
    num_prepro_workers=8,
    enable_text=True,
    enable_image=True,
    enable_metadata=False,
    enable_unclip=False,
    enable_vae=False,
    write_batch_size=10**6,
    wds_image_key="jpg",
    wds_caption_key="txt",
    clip_model="ViT-B/32",
    mclip_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
    unclip_model="",
    vae_model="",
    use_mclip=False,
    use_jit=True,
    distribution_strategy="sequential",
    wds_number_file_per_input_file=10000,
    output_partition_count=None,
    wandb_project="clip_retrieval",
    enable_wandb=False,
):
    if enable_unclip:
        assert unclip_model != "", "unclip_model must be set if enable_unclip is True"
        assert enable_text, "enable_text must be True if enable_unclip is True"
        assert (
            clip_model == "ViT-L/14"
        ), "ViT-L/14 is the only supported model for unclip"
        assert not use_mclip, "You cannot use mclip with unclip"
        assert (
            len(unclip_model) > 0
        ), "You must specify a diffusion conditioned prior model when `enable_unclip` is True"
    if enable_vae:
        assert vae_model != "", "vae_model must be set if enable_vae is True"
        assert Path(vae_model).exists(), "vae_model must exist"
        assert enable_image, "enable_image must be True if enable_vae is True"

    if input_format == "webdataset":
        if "*.tar" in input_dataset:
            input_dataset = glob(input_dataset)
        else:
            input_dataset = list(braceexpand(input_dataset))
            print(
                f"found {len(input_dataset)} input files from webdataset at {input_dataset}"
            )  # TODO
    if output_partition_count is None:
        if input_format == "files":
            keys, text_files, image_files, metadata_files = folder_to_keys(
                input_dataset,
                enable_text=enable_text,
                enable_image=enable_image,
                enable_metadata=enable_metadata,
            )
            if text_files is None or len(text_files) == 0:
                enable_text = False
            if image_files is None or len(image_files) == 0:
                enable_image = False
            if metadata_files is None or len(metadata_files) == 0:
                enable_metadata = False
            keys, text_files, image_files, metadata_files = folder_to_keys(
                input_dataset,
                enable_text=enable_text,
                enable_image=enable_image,
                enable_metadata=enable_metadata,
            )
            sample_count = len(keys)
        elif input_format == "webdataset":
            sample_count = len(input_dataset) * wds_number_file_per_input_file
        else:
            print("Unsupported input_format")
            return

        if sample_count == 0:
            print("no sample found")
            return
        else:
            print(f"The number of samples has been estimated to be {sample_count}")

        output_partition_count = int(sample_count / write_batch_size) + 1

    def reader_builder(sampler):
        
        if enable_vae:
            preprocess = vae_preprocess
        else:
            preprocess = clip_transform(224)
        if input_format == "files":
            return FilesReader(
                sampler=sampler,
                preprocess=preprocess,
                input_dataset=input_dataset,
                batch_size=batch_size,
                num_prepro_workers=num_prepro_workers,
                enable_text=enable_text,
                enable_image=enable_image,
                enable_metadata=enable_metadata,
            )
        elif input_format == "webdataset":
            return WebdatasetReader(
                sampler=sampler,
                preprocess=preprocess,
                input_dataset=input_dataset,
                batch_size=batch_size,
                num_prepro_workers=num_prepro_workers,
                enable_text=enable_text,
                enable_image=enable_image,
                enable_metadata=enable_metadata,
                wds_image_key=wds_image_key,
                wds_caption_key=wds_caption_key,
                cache_path=cache_path,
            )
        else:
            raise ValueError(f"Unknown input_format: {input_format}")

    def mapper_builder():
        return ClipMapper(
            enable_image=enable_image,
            enable_text=enable_text,
            enable_unclip=enable_unclip,
            enable_vae=enable_vae,
            enable_metadata=enable_metadata,
            use_mclip=use_mclip,
            clip_model=clip_model,
            use_jit=use_jit,
            mclip_model=mclip_model,
            unclip_model=unclip_model,
            vae_model=vae_model,
        )

    def writer_builder(i):
        return NumpyWriter(
            partition_id=i,
            output_folder=output_folder,
            enable_text=enable_text,
            enable_image=enable_image,
            enable_unclip=enable_unclip,
            enable_vae=enable_vae,
            enable_metadata=enable_metadata,
            output_partition_count=output_partition_count,
        )

    def logger_builder(i):
        return LoggerWriter(
            partition_id=i,
            stats_folder=output_folder + "/stats",
        )

    runner = Runner(
        reader_builder=reader_builder,
        mapper_builder=mapper_builder,
        writer_builder=writer_builder,
        logger_builder=logger_builder,
        output_partition_count=output_partition_count,
    )

    logger_reader = LoggerReader(
        stats_folder=output_folder + "/stats",
        wandb_project=wandb_project,
        enable_wandb=enable_wandb,
    )
    logger_reader.start()

    if distribution_strategy == "sequential":
        distributor = SequentialDistributor(runner, output_partition_count)
    elif distribution_strategy == "pyspark":
        distributor = PysparkDistributor(runner, output_partition_count)
    distributor()

    logger_reader.end()


if __name__ == "__main__":
    fire.Fire(main)
