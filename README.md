# IVP_mildew_classification
Image and Video Processing project at Strathclyde that aims to classify potato leaves for mildew detection.

# Project initialisation

### WSL and Visual Studio Code installation

In a Windows powershell, run

```
wsl --install
```

Then install Visual Studio code on Windows using this link : https://code.visualstudio.com/download

When prompted to Select Additional Tasks during installation, be sure to check the Add to PATH option so you can easily open a folder in WSL using the code command (Je me rappelle pas de l'avoir fait mais si vous le voyez faites le, ça mange pas de pain)

Open WSL in a powershell
```
wsl
```
Update Linux 
```bash
sudo apt-get update
```
To add wget (to retrieve content from web servers) and ca-certificates (to allow SSL-based applications to check for the authenticity of SSL connections):
(Je sais pas si cette ligne est utile mais ils disent de le faire)
```bash
sudo apt-get install wget ca-certificates
```

## Clone project

Ouvrir un powershell WSL à partir d'un powershell windows

```bash
wsl
```

Je me rappelle plus si j'ai eu besoin d'installer git sur wsl, je crois que non. Si besoin, j'imagine que vous trouverez votre bonheur ici: https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-git


Clone project
```bash
git clone https://github.com/JulienPriam/IVP_mildew_classification.git
```

## Open project in VSCode from WSL

To open VSCode from WSL, the WSL extension is needed in VSCode. To install it, go to Extensions on the left in VSCode (or `CTRL+Shift+X`), search for WSL and install Extension.

Then to open the project in VSCode, go to the project directory in the WSL prompt and run
```bash
code .
```

Bravo, ça fonctionne normalement, et sinon Google est votre ami (:

Btw quelques extensions pratiques à installer dans VSCode :
`Python` (pour notamment pouvoir run des trucs facilement)
`Markdown all in one`
`Markdown Preview enhance` (pour avoir une jolie preview du markdown)

Pour ouvrir la preview quand on est sur un fichier markdown : `CTRL+k puis v`

## Packages installation

Perso j'ai utilisé un virtual env avec pip, vous faites comme vous préférez.

##### Install python and pip

To install python
```bash
sudo apt-get install python3
sudo apt-get install python3-venv
```

##### Install needed packages

Create a virtual environment in your project

```bash
python3 -m venv IVP_env
```
Please prenez le même nom que moi

Activate it

```bash
source venv_name/bin/activate
```
Install packages needed by running
```bash
pip install opencv-python-headless numpy matplotlib
```

##### Select interpreter in VSCode

`CTRL+Shift+P` then search for `Python: Select Interpreter` in the prompt. Choose `./IVP_env/bin/python`.

## Download dataset

In `/home/your_name/IVP_mildew_classification`, create a folder `potato_dataset`.

Download the dataset from this link
https://www.kaggle.com/datasets/arjuntejaswi/plant-village?resource=download
Extract the dataset from the downloaded zip. Rename the three folers containing potato leaves

Potato___Early_blight --> early_blight
Potato___Late_blight --> early_blight
Potato___healthy --> healthy

Then copy these three folders. Dans la barre d'exploration, taper `\\wsl$`, puis chercher le folder du projet, et coller les 3 folders dans le folder `potato_dataset`

Then you can run the code !