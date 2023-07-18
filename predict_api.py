import paddleseg

model = "null"
transforms = "null"
model_path = "null"
image_list  = "null"

paddleseg.core.predict(model,
                       model_path,
                       transforms,
                       image_list,
                       image_dir=None,
                       save_dir='output',
                       aug_pred=False,
                       scales=1.0,
                       flip_horizontal=True,
                       flip_vertical=False,
                       is_slide=False,
                       stride=None,
                       crop_size=None
)

