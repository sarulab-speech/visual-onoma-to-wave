mkdir -p outputs/RWCP-SSD/latest/ckpt
File_id="1iklfUAU2CSuMaOdxFUQgyeA-wZF1XnI1"
File_name="200000.pth.tar"
gdown --id ${File_id} -O outputs/RWCP-SSD/latest/ckpt/${File_name}
unzip scripts/hifigan/generator_universal.pth.tar.zip -d scripts/hifigan/