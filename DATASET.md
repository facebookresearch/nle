# Dungeon And Data

Instructions and Tutorials for the NetHack Learning Dataset

## License

This data is licensed under the NetHack General Public License - based on the GPL-style BISON license. It is the license used for the game of NetHack, and can be found [here](https://github.com/facebookresearch/nle/blob/main/LICENSE).


## Accessing the Dataset

The dataset is currently hosted on FAIR's AWS with open access for all. Given its large size it has been split into smaller chunks for ease of downloading. 


### Download Links

#### TasterPacks

We provide a small "taster pack" dataset, that contain a random subsample of the full datasets, to allow fast iteration for those looking to play around with `nld`.
- [`nld-aa-taster.zip`](https://dl.fbaipublicfiles.com/nld/nld-aa-taster/nld-aa-taster.zip
) (1.6GB)


#### Full Downloads
You can download these by visiting the links or using curl:

`NLD-AA` (16 file)
```
# Download NLD-AA
mkdir -p nld-aa
curl -o nld-aa/nld-aa-dir-aa.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-aa.zip
curl -o nld-aa/nld-aa-dir-ab.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ab.zip
curl -o nld-aa/nld-aa-dir-ac.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ac.zip
curl -o nld-aa/nld-aa-dir-ad.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ad.zip
curl -o nld-aa/nld-aa-dir-ae.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ae.zip
curl -o nld-aa/nld-aa-dir-af.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-af.zip
curl -o nld-aa/nld-aa-dir-ag.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ag.zip
curl -o nld-aa/nld-aa-dir-ah.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ah.zip
curl -o nld-aa/nld-aa-dir-ai.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ai.zip
curl -o nld-aa/nld-aa-dir-aj.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-aj.zip
curl -o nld-aa/nld-aa-dir-ak.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ak.zip
curl -o nld-aa/nld-aa-dir-al.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-al.zip
curl -o nld-aa/nld-aa-dir-am.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-am.zip
curl -o nld-aa/nld-aa-dir-an.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-an.zip
curl -o nld-aa/nld-aa-dir-ao.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ao.zip
curl -o nld-aa/nld-aa-dir-ap.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ap.zip
```



`NLD_NAO` (41 files)
```
# Download NLD-NAO
curl -o nld-nao/nld-nao-dir-aa.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-aa.zip
curl -o nld-nao/nld-nao-dir-ab.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ab.zip
curl -o nld-nao/nld-nao-dir-ac.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ac.zip
curl -o nld-nao/nld-nao-dir-ad.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ad.zip
curl -o nld-nao/nld-nao-dir-ae.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ae.zip
curl -o nld-nao/nld-nao-dir-af.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-af.zip
curl -o nld-nao/nld-nao-dir-ag.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ag.zip
curl -o nld-nao/nld-nao-dir-ah.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ah.zip
curl -o nld-nao/nld-nao-dir-ai.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ai.zip
curl -o nld-nao/nld-nao-dir-aj.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-aj.zip
curl -o nld-nao/nld-nao-dir-ak.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ak.zip
curl -o nld-nao/nld-nao-dir-al.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-al.zip
curl -o nld-nao/nld-nao-dir-am.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-am.zip
curl -o nld-nao/nld-nao-dir-an.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-an.zip
curl -o nld-nao/nld-nao-dir-ao.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ao.zip
curl -o nld-nao/nld-nao-dir-ap.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ap.zip
curl -o nld-nao/nld-nao-dir-aq.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-aq.zip
curl -o nld-nao/nld-nao-dir-ar.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ar.zip
curl -o nld-nao/nld-nao-dir-as.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-as.zip
curl -o nld-nao/nld-nao-dir-at.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-at.zip
curl -o nld-nao/nld-nao-dir-au.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-au.zip
curl -o nld-nao/nld-nao-dir-av.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-av.zip
curl -o nld-nao/nld-nao-dir-aw.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-aw.zip
curl -o nld-nao/nld-nao-dir-ax.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ax.zip
curl -o nld-nao/nld-nao-dir-ay.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ay.zip
curl -o nld-nao/nld-nao-dir-az.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-az.zip
curl -o nld-nao/nld-nao-dir-ba.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-ba.zip
curl -o nld-nao/nld-nao-dir-bb.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bb.zip
curl -o nld-nao/nld-nao-dir-bc.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bc.zip
curl -o nld-nao/nld-nao-dir-bd.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bd.zip
curl -o nld-nao/nld-nao-dir-be.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-be.zip
curl -o nld-nao/nld-nao-dir-bf.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bf.zip
curl -o nld-nao/nld-nao-dir-bg.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bg.zip
curl -o nld-nao/nld-nao-dir-bh.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bh.zip
curl -o nld-nao/nld-nao-dir-bi.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bi.zip
curl -o nld-nao/nld-nao-dir-bj.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bj.zip
curl -o nld-nao/nld-nao-dir-bk.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bk.zip
curl -o nld-nao/nld-nao-dir-bl.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bl.zip
curl -o nld-nao/nld-nao-dir-bm.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bm.zip
curl -o nld-nao/nld-nao-dir-bn.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bn.zip
curl -o nld-nao/nld-nao_xlogfiles.zip https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao_xlogfiles.zip
```

### Reconstructing the Dataset

Unzip the files in the standard way, with separate directories for `NLD-AA`, and `NLD-NAO`. 


```bash
# for NLD-AA
# will give you an nle_data directory at /path/to/dir/nld-aa-dir/nld-aa/nle_data/
$ unzip /path/to/nld-aa/nld-aa-dir-aa.zip  -d /path/to/dir
$ unzip /path/to/nld-aa/nld-aa-dir-ab.zip  -d /path/to/dir
$ unzip /path/to/nld-aa/nld-aa-dir-ac.zip  -d /path/to/dir
...


# for NLD-NAO - don'f forget xlogfiles
# will give you an directory with ending with nld-nao-unzipped/
$ unzip /path/to/nld-xlogfiles.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao-dir-aa.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao-dir-ab.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao-dir-ac.zip -d /path/to/nld-nao
...
```


## Using the Dataset ([Colab Demo](https://colab.research.google.com/drive/1GRP15SbOEDjbyhJGMDDb2rXAptRQztUD?usp=sharing))

The code needed to use the dataset will be distributed in `NLE v0.9.0`. For now it can be found on the `main` branch of [NLE](https://github.com/facebookresearch/nle). You can follow the instructions to install [there](https://github.com/facebookresearch/nle), or try the below.

```
# With pip:
pip install git+https://github.com/facebookresearch/nle.git@main

# From source:
git clone --recursive https://github.com/facebookresearch/nle.git
cd nle && pip install -e .
```

Once this is installed, you simply need to load the `nld` folders (once) which will create a small local sqlite3 database, and then you can use the dataset.

```python
import nle.dataset as nld

if not nld.db.exists():
    nld.db.create()
    # NB: Different methods are used for data based on NLE and data from NAO.
    nld.add_nledata_directory("/path/to/nld-aa/nle_data", "nld-aa-v0")
    nld.add_altorg_directory("/path/to/nld-nao-unzipped", "nld-nao-v0")

dataset = nld.TtyrecDataset("nld-aa-v0", batch_size=128, ...)
for i, mb in enumerate(dataset):
    foo(mb) # etc...
```

for more instructions on usage see the accompanying [Colab notebook](https://colab.research.google.com/drive/1GRP15SbOEDjbyhJGMDDb2rXAptRQztUD?usp=sharing).


## Troubleshooting

If you are having issues loading the dataset, ensure that the directory structure is as laid out in the docstrings to the `add_*_directory` functions.

``` python
help(nld.add_nledata_directory) # will print docstring 
```
