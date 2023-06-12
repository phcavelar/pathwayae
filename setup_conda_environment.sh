env_name=pae
# Check for cuda capabilities
if command -v nvidia-smi; then
    if nvidia-smi | grep 'CUDA Version: 11.[3456]'; then
        install_type='cudatoolkit=11.3'
        pytorch_type='pytorch==1.12'
    elif nvidia-smi | grep 'CUDA Version: 11.[789]'; then
        install_type='pytorch-cuda=11.7'
        pytorch_type='pytorch==1.13'
        extra_nvidia_channel=' -c nvidia '
    elif nvidia-smi | grep 'CUDA Version: 1[23456789].'; then
        install_type='pytorch-cuda=11.7'
        pytorch_type='pytorch==1.13'
        extra_nvidia_channel=' -c nvidia '
    elif nvidia-smi | grep 'CUDA Version: 11.[012]'; then
        install_type='cudatoolkit=10.2'
        pytorch_type='pytorch==1.12'
    elif nvidia-smi | grep 'CUDA Version: 10.[23456789]'; then
        install_type='cudatoolkit=10.2'
        pytorch_type='pytorch==1.12'
    else
        install_type='cpuonly'
    fi
else
    install_type='cpuonly'
fi

conda create --force -n ${env_name}

if command -v mamba; then
    if echo ${install_type} | grep 'cpuonly'; then
        # Mamba fails to solve cpuonly pyg https://github.com/mamba-org/mamba/issues/1542
        install_method='conda'
    else
        install_method='mamba'
    fi
else
    install_method='conda'
fi

command ${install_method} install -n ${env_name} -c pytorch ${extra_nvidia_channel} -c conda-forge ${pytorch_type} ${install_type} numpy scipy pandas matplotlib seaborn ipykernel ipywidgets tqdm fire pytorch-lightning scikit-learn

echo "done"
