import os
import tensorflow as tf
import nibabel as nib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from utils_brats import preprocess, CLASSES

slice_num = 60
cur_dirr = os.path.abspath(os.getcwd())


def predict_tumors(X, dropdown):

    pred = tf.squeeze(unet.predict(tf.expand_dims(X, 0)/tf.reduce_max(X)), 0)

    plt.imshow(X[..., 0], cmap='gray', alpha=0.3)

    if dropdown == CLASSES[1]:
        plt.imshow(pred[..., 1], cmap="OrRd", alpha=0.7)
        plt.title(f'{CLASSES[1]} predicted')
        plt.axis(False)

    elif dropdown == CLASSES[2]:
        plt.imshow(pred[..., 2], cmap="OrRd", alpha=0.7)
        plt.title(f'{CLASSES[2]} predicted')
        plt.axis(False)

    elif dropdown == CLASSES[3]:
        plt.imshow(pred[..., 3], cmap="OrRd", alpha=0.7)
        plt.title(f'{CLASSES[3]} predicted')
        plt.axis(False)

    else:
        plt.imshow(pred[..., 1:4], cmap="OrRd", alpha=0.7)
        plt.title("Predicted")
        plt.axis(False)

    plt.tight_layout()
    plt.savefig(cur_dirr + '\\static\\out_img.png')
    # plt.show()


menuu = list(CLASSES.values())[1:]

unet = tf.keras.models.load_model("2dunet_vir.h5", compile=False)
print('\nready to launch...\n')


app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html", menuu=menuu)


@app.route('/outpg', methods=["GET", "POST"])
def outtputs():

    flair_path = os.path.abspath(request.form.get("flair_path"))
    ce_path = os.path.abspath(request.form.get("ce_path"))
    select_tumor = (request.form.get("part_of_tumor"))

    if (not flair_path):
        return render_template("failure.html", message="missing flair path")
    if (not ce_path):
        return render_template("failure.html", message="missing ce path")
    if (select_tumor not in menuu and select_tumor != 'all_the_tumor'):
        return render_template("failure.html", message="invalid tumor part")


    flair = nib.load(flair_path).get_fdata()
    ceimg = nib.load(ce_path).get_fdata()

    x = preprocess(flair=flair, ce=ceimg, for_pred=True)
    x = tf.transpose(x, [2, 0, 1, 3])[slice_num]

    plt.imshow(x[..., 0], cmap='gray')
    plt.title('flair input image')
    plt.axis(False)
    plt.tight_layout()
    plt.savefig(cur_dirr + '\\static\\input1.png')

    plt.imshow(x[..., 1], cmap='gray')
    plt.title('ce input image')
    plt.axis(False)
    plt.tight_layout()
    plt.savefig(cur_dirr + '\\static\\input2.png')

    predict_tumors(X=x, dropdown=select_tumor)

    return render_template("success.html",
                           flair_path="input1.png",
                           ce_path = "input2.png",
                           output_path="out_img.png")


if __name__ == '__main__':
    app.run(debug=True)
