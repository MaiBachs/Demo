from PIL import Image

# Mở hình ảnh cần chuyển sang mức xám
image = Image.open('1.png')

# Chuyển hình ảnh sang mức xám
image = image.convert('L')

# Lưu hình ảnh chuyển đổi vào tệp mới
image.save('1ed.png')
