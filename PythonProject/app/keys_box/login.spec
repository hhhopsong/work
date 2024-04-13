# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['login.py', 'C:\\Users\\admin\\PycharmProjects\\pythonProject1\\app\\keys_box\\main_window.py', 'C:\\Users\\admin\\PycharmProjects\\pythonProject1\\app\\keys_box\\view.py'],
             pathex=[],
             binaries=[],
             datas=[('C:\\Users\\admin\\PycharmProjects\\pythonProject1\\app\\keys_box\\key.txt', '.'), ('C:\\Users\\admin\\PycharmProjects\\pythonProject1\\app\\keys_box\\data.txt', '.')],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='login',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None,
          icon='C:\\Users\\admin\\PycharmProjects\\pythonProject1\\app\\keys_box\\favicon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='login')
