from PIL import Image
import base64
import io


def from_image_to_b64(path):
    image = Image.open(path)
    output = io.BytesIO()
    image.save(output, format='PNG')
    encoded_string = "data:image/jpeg;base64," + \
                     base64.b64encode(output.getvalue()).decode()
    return encoded_string


def get_image(path):
    return Image.open(path)
