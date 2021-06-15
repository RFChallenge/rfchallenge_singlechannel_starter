# Single-Channel RF Challenge Starter Code

The helper functions in this starter code use SigMF and CommPy. To install these dependencies:
```bash
pip install git+https://github.com/gnuradio/SigMF.git
pip install scikit-commpy
```
Other dependencies include: NumPy, Matplotlib, Tensorflow (to run the [`bitregression`](https://github.com/RFChallenge/rfchallenge_singlechannel_starter/tree/main/example/demod_bitregression) example), tqdm (for progress bar)  

Please ensure that the `dataset` is saved in the folder "dataset", and retains the folder hierarchy as provided -- this allows the helper functions to correctly find and load the corresponding files.

To obtain the dataset, you may use the following commands:
```bash
wget -O rfc_dataset.zip "https://www.dropbox.com/s/clh4xq7u3my3nx6/rfc_dataset.zip?dl=0"
unzip -o rfc_dataset.zip
rm rfc_dataset.zip
```


The python notebook [`notebook/Demo.ipynb`](https://github.com/RFChallenge/rfchallenge_singlechannel_starter/blob/main/notebook/Demo.ipynb) demonstrates how these helper functions may be used to load the respective sigmf files from the training and validation datasets.

The python notebook [`notebook/QuickStart.ipynb`](https://github.com/RFChallenge/rfchallenge_singlechannel_starter/blob/main/notebook/QuickStart.ipynb) provides a brief overview on helper functions and code snippet that will get you started.

Refer to the python notebook [`notebook/Reference_Methods.ipynb`](https://github.com/RFChallenge/rfchallenge_singlechannel_starter/blob/main/notebook/Reference_Methods.ipynb) for reference methods that you can compare against!

---

### Note regarding examples:
As some of the saved files in the example methods are large (e.g. saved models and statistics), they are not included in this Git repository. 
You can get get the full contents of the `example` folder using the following commands:
```bash
wget -O rfc_example.zip "https://www.dropbox.com/s/mlwlhnouz4ljdly/rfc_example.zip?dl=0"
unzip -o rfc_example.zip
rm rfc_example.zip
```
NB: `unzip -o` overwrites files in the example folder. If you have modified or added files to the example folder, back up those files before running the above commands.

This step is only required if you wish to run scripts or functions provided in the `example` folder.

---
### Direct Download Links:
* [Dataset (Training set)](https://www.dropbox.com/s/clh4xq7u3my3nx6/rfc_dataset.zip?dl=0)
* [Example folder from starter code](https://www.dropbox.com/s/mlwlhnouz4ljdly/rfc_example.zip?dl=0)
