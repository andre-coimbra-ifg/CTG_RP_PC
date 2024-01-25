# CTG_RP_PC
ML/DL analysis of Cardiotocography (CTG) traces using Recurrence Plot and Poincaré Plot.

This repo was forked from [Williams Doug's CTG_RP](https://github.com/williamsdoug/CTG_RP) repo.

The original repo contained Jupyter Notebooks and code to reproduce the results of the paper [Computer-Aided Diagnosis System of Fetal Hypoxia Incorporating Recurrence Plot With Convolutional Neural Network](https://www.frontiersin.org/articles/10.3389/fphys.2019.00255/full)
 by Zhidong Zhao, Yang Zhang, Zafer Comert and Yanjun Deng.

The Notebooks and the code were adapted to check the results with Recurrence Plot and Poincaré Plot. The images were saved in the TIFF format, using the DEFLATE compression scheme, considering the paper [Compression of Different Time Series Representations in Asphyxia Detection](https://ieeexplore.ieee.org/iel7/9991246/9991267/09991468.pdf)
 by Bárbara Silva, Maria Ribeiro and Teresa S. Henriques.

This repo was made for the final project requirements of the course Special Topics in Computer Systems, Graduate Program in Computer Science, UFG, 2023/2.

## Implementation Details

Key Jupyter Notebooks (_currently configured to run on Google Colab per below_):
- [CTG_RP_ResNet](https://github.com/andre-coimbra-ifg/CTG_RP_PC/blob/master/CTG_RP_ResNet.ipynb)
  - Initializes fresh colab instance, downloading source files, packages and dataset
  - Generates RP images
    -  RP Images based on latest valid 10min CTG segment
  -  CTG recordings partitioned into Train and Valid
  -  Trains FastAI Model
     -  Uses transfer learning to train using ResNetX [18, 34, 50] model
 
  

- [CTG_PC_ResNet](https://github.com/andre-coimbra-ifg/CTG_RP_PC/blob/master/CTG_PC_ResNet.ipynb)
  - Initializes fresh colab instance, downloading source files, packages and dataset
  - Generates PC images
    -  PC Images based on latest valid 10min CTG segment
  -  CTG recordings partitioned into Train and Valid
  -  Trains FastAI Model
    -  Uses transfer learning to train using ResNetX [18, 34, 50] model
 
Other Notebooks:
- [CTG_RP_Display_Denoised](https://github.com/andre-coimbra-ifg/CTG_RP_PC/blob/master/CTG_RP_Display_Denoised.ipynb)
  - Displays sample denoised signals
  
- [CTG_RP_Generate_Recurrence_Plots](https://github.com/andre-coimbra-ifg/CTG_RP_PC/blob/master/CTG_RP_Generate_Recurrence_Plots.ipynb)
  - Creates individual RP Images.  _IMAGES_DIR/rp_images_index.json_ contains metadata associated with images for each recording
  
- [CTG_RP_Explore_Datasets](https://github.com/andre-coimbra-ifg/CTG_RP_PC/blob/master/CTG_RP_Explore_Datasets.ipynb)
  - Builds Databunch and displays contents


## Key Dependencies

- Data: 
  - [CTU-UHB Intrapartum Cardiotocography Database](https://physionet.org/physiobank/database/ctu-uhb-ctgdb/)
    - Paper: [Open access intrapartum CTG database](https://bmcpregnancychildbirth.biomedcentral.com/track/pdf/10.1186/1471-2393-14-16)
    - Download: 
      - `rsync -Cavz physionet.org::ctu-uhb-ctgdb  /content/ctu-uhb-ctgdb`

- Libraries:
  - [pyts](https://pyts.readthedocs.io/en/latest/)
    - Used for generation of Recurrence Plots
    - `pip install pyts`
  - [FastAI](https://docs.fast.ai/) library running on [PyTorch](https://pytorch.org/)
    - Used for deep learning
    - Installed by default on Google Colab
    - Version: 2.7.13
  - [wfdb](https://wfdb.readthedocs.io/en/latest/index.html)
    - Waveform Database Utilities.  Used to read Physionet Recording Files
    - `pip install wfdb`
    
- Computation:
  - [Google Colab](https://colab.research.google.com)
    - Jupyter Notebook service with no-cost access to GPU accelerated instances
