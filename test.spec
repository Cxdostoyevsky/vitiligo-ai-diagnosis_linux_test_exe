# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

# --- datas:打包程序运行的时候的非代码资源 ---
datas = [('model_ckpt', 'model_ckpt')] + collect_data_files('transformers')

# --- 新增: 更全面的隐藏导入列表，解决 "Unrecognized class" 问题 ---
hiddenimports = [
    # Existing
    'transformers',
    'safetensors',
    'safetensors.torch',
    'tokenizers',
    'tokenizers.implementations',
    'tokenizers.models',
    'tokenizers.pre_tokenizers',
    'tokenizers.processors',
    'tokenizers.decoders',
    'tokenizers.normalizers',
    'pydantic',
    'pydantic_core',
    
    # For Siglip (already present, but good to keep track)
    'transformers.models.siglip',
    'transformers.models.siglip.modeling_siglip',
    'transformers.models.siglip.processing_siglip',
    'transformers.models.siglip.image_processing_siglip',
    'transformers.models.siglip.tokenization_siglip',
    'transformers.models.siglip.configuration_siglip',

    # For Auto* classes
    'transformers.models.auto.processing_auto',
    'transformers.models.auto.modeling_auto',
    'transformers.models.auto.image_processing_auto',
    'transformers.models.auto.tokenization_auto',
    'transformers.models.auto.configuration_auto',

    # New for LVLM
    'transformers.models.llama',
    'transformers.models.llama.modeling_llama',
    'transformers.models.llama.configuration_llama',
    'transformers.models.qwen',
    'transformers.models.qwen.modeling_qwen2', # Assuming Qwen2
    'transformers.models.qwen.configuration_qwen2',
    'transformers.models.phi',
    'transformers.models.phi.modeling_phi',
    'transformers.models.phi.configuration_phi',
    'transformers.models.phi3',
    'transformers.models.phi3.modeling_phi3',
    'transformers.models.phi3.configuration_phi3',
    'transformers.models.clip',
    'transformers.models.clip.modeling_clip',
    'transformers.models.clip.processing_clip',
    'transformers.models.clip.image_processing_clip',
    'transformers.models.clip.configuration_clip',
    'transformers.models.bert',
    'transformers.models.bert.modeling_bert',
    'transformers.models.bert.tokenization_bert',
    'transformers.models.bert.configuration_bert',
    'monai',
    'einops',
]

a = Analysis(
    ['test.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports, # <--- 使用我们新的、更完整的列表
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='test', # <--- 生成的可执行文件名
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='test_dist', # <--- 包含可执行文件和所有依赖的文件夹名称
)
