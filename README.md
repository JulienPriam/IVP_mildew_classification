# IVP_mildew_classification
Image and Video Processing project at Strathclyde that aims to classify potato leaves for mildew detection.

# Project initialisation

## Clone project

Je me rappelle plus si j'ai eu besoin d'installer git. Si besoin, j'imagine que vous trouverez votre bonheur ici: https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-git

Create a directory for the project
```bash
mkdir project_dir
```
Clone project
```bash
git clone https://github.com/JulienPriam/IVP_mildew_classification.git
```
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

### Packages installation

Perso j'ai utilisé un environnement conda parce que c'est pratique, vous pouvez utiliser un virtual env avec pip si vous préférer, mais je sais pas comment on fait. Du coup je vous donne le tuto pour Conda.

##### Install Conda

If you use Debian, this prerequisite is needed:

```bash
apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```

If you use Ubuntu, nothing needed as prerequisites (Si WSL, c'est Ubuntu)

Install Anaconda
```bash
wget https://repo.continuum.io/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh
conda update -n base -c defaults conda
```
Close and reopen terminal

##### Install needed packages

First create a conda environment
```bash
conda create -n IVP
```

