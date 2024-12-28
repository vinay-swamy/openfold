module load rocm/6.2.1
#module load rocm/5.7.1
mamba env create -n amdof_relaxhip -f environment_amd_hiprelax.yml
mamba activate amdof_relaxhip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install pytorch-lightning==2.1.4 openmm==8.2.0
pip install openmm[hip6]
python setup.py install 



# #test 
# python3 run_pretrained_openfold.py test_input/ \
#   /p/vast1/OpenFoldCollab/openfold-data/pdb_mmcif/mmcif_files/ \
#   --output_dir test/ \
#   --use_precomputed_alignments test_input/ \
#   --config_preset model_3 \
#   --openfold_checkpoint_path /p/vast1/OpenFoldCollab/openfold-amd/openfold_params/finetuning_no_templ_2.pt \
#   --model_device "cuda"