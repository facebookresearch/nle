# Dungeon And Data

Instructions and Tutorials for the NetHack Learning Dataset

## License

This data is licensed under the NetHack General Public License - based on the GPL-style BISON license. It is the license used for the game of NetHack, and can be found [here](https://github.com/facebookresearch/nle/blob/main/LICENSE).


## Accessing the Dataset

The dataset is currently hosted on WeTransfer with open access for all. It will eventually move to its own dedicated hosting site, which is in the process of being set up. For the time being, `NLD-AA` is one file, while `NLD-NAO` is in 5 parts (4 ttyrec zips + the xlogfiles).


### Download Links

#### TasterPacks

We provide a small "taster pack" dataset, that contain a random subsample of the full datasets, to allow fast iteration for those looking to play around with `nld`.
- [`nld-aa-taster.zip`](https://we.tl/t-YjbQy4yPQJ) (1.6GB)


#### Full Downloads

`NLD-AA` (1 file)
- [`nld-aa.zip`](https://we.tl/t-wwN4lD7Hqn) (90 GB)


`NLD_NAO` (5 files)
- [`nld-nao_part1.zip`](https://we.tl/t-XQe15aXAes) (54GB)
- [`nld-nao_part2.zip`](https://we.tl/t-YRHHAb9gTe) (63GB)
- [`nld-nao_part3.zip`](https://we.tl/t-XB0iundCAU) (54GB)
- [`nld-nao_part4.zip`](https://we.tl/t-pkWlT0yTFK) (42GB)
- [`nld-nao_xlogfiles.zip`](https://we.tl/t-vy7IAGohCu) (124MB)


#### Downloading from Command Line

WeTransfer obscures the final download link, appending authentication keys to the link.  To obtain a final working url that can function with, for instance, `wget` or `curl`:

**Firefox**

1. Start a download as usual, then cancel it. 
2. Open Downloads (⌘J)
3. Right-click on the Download and click "Copy Download Link"

**Chrome**
1. Start a download as usual, then cancel it. 
2. Open Downloads (⇧⌘J)
3. Right-click on the Link and click "Copy Link Address"

### Reconstructing the Dataset

Unzip the files in the standard way, with separate directories for `NLD-AA`, and `NLD-NAO`. 


```bash
$ unzip /path/to/nld-aa.zip 

$ unzip /path/to/nld-xlogfiles.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao_part1.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao_part2.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao_part3.zip -d /path/to/nld-nao
$ unzip /path/to/nld-nao_part4.zip -d /path/to/nld-nao
```


- NB: `NLD-AA` is already a single directory, so will unzip to one directory already,
where as all the `NLD-NAO` files should be zipped to one directory.

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
    nld.add_nledata_directory("/path/to/nld-aa", "nld-aa-v0")
    nld.add_altorg_directory("/path/to/nld-nao", "nld-nao-v0")

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
