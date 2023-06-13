# Class Based Style Transfer
## Fast Style Transfer
The fast style transfer code can be trained on custom styles and datasets with a single line command.
        
>python main.py train -train _path\_to\_training\_images_ -styles _path\_to\_style\_image_

>python main.py train -train _path\_to\_training\_images_ -styles _path\_to\_style\_image_ -single_style

>python main.py style -model _path\_to\_model_ -tostyle _path\_to\_images_

>python main.py -image_path ./datasets/citys/leftImg8bit/train/bremen/bremen_000001_000019_leftImg8bit.png