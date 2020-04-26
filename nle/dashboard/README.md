# NetHack Dashboard

This dashboard allows to load NetHack runs and visualize the recording of the 
various episodes. You can also sort runs by some parameter, step through them,
view agent actions distribution and more.

The dashboard is a nodeJS application, which makes it portable and easy to use.


## Installation and usage

To use this dashboard you need to install nodeJS. In a `conda` environment, you
can do so by doing: `conda install nodejs`.

Once you have nodeJS installed, move to `NetHack/dash` (this folder) and 
start the server:
```
npm install # Only the first time, installs the required dependencies.
npm start # Starts the server.
```

By default, the server listens on `localhost:3000` (you can change this from
`config.js`). In order to see the dashboard, just connect to it from your
preferred browser.

Once you are in the dashboard, insert the path to the folder containing the
data you want to load and press the load button. You can also add the `path`
argument to the URL, and this will automatically load your data when the
dashboard is being loaded. For example:
```
http://localhost:3000/dashboard.html?path=/path/to/your/data
```
If you pass a relative path (e.g. `path/to/your/data` instead of
`/path/to/your/data`) the server will add the asbolute path to the `dash/`
folder as a prefix to your path.

You can also make the server recursively explore subfolders of the path you
provide and load all the relevant data it can find.


## Data format
When loading data from a folder, the server will look for a single file called
`stats.csv` (you can change this from `config.js`).
This is a csv file with the information about the NetHack runs
to load, one for each line. This csv file should have no header, since the
fields are harcoded in `config.js`. Note that if the number of header fields
does not match the number actual fields in the csv file, you won't be able
to load the data. (TODO: change this to automatically get the csv header).

The csv header must contain at least the field `ttyrec`, which is used by the
client in order to fetch the files with the recordings (this can be changed at
the top of `dashboard.html`). The field must be present but can be empty,
which means no recording has been kept for that run.

An example can be found in `data/`.


## Overview of the fetching process

The fetching process has 6 main steps:

1) The server is started and the client connects to it.
2) The client sends a `/runs_info` request to the server, with the path
to the folder containing the desired `stats.csv` file.
3) The server accesses the `stats.csv` in the specified folder and
retreives the available information, which are then sent back to the
client.
4) The client makes sure the information are valid (i.e. they are non-empty and
contain all the required fields).
5) The client sends a bunch of `/ttyrec_file` requests to the server, in order
to fetch the relevant ttyrec files. Each request contains the path to the file
to fetch (which is taken from the runs info previously fetched).
6) The server compresses and sends the requested ttyrec files.

If errors are encountered during the process, they are (usually) surfaced to
the user.
In case of problems, it is a good idea to check both the server logs and the
browser console.


## Repository content
- `app/`: this folder contains the files loaded by the client. It contains:
  - `third_party/`: contains third party libraries. Note that even though
  `ttyplay.js` is in this folder, it has been heavily modified in order to serve
  our porpuse. All libraries are covered by the MIT licence. `ttyplay` was
  originally taken from [oliy/ttyplay](https://github.com/oliy/ttyplay).
  `apexcharts.js` instead was taken from
  [apexcharts/apexcharts.js](https://github.com/apexcharts/apexcharts.js).
  - `dashboard.html`: the actual dashboard page, with a minimal html and the
  JS functions to fetch, elaborate and display data.
  - `style.css`: the css for `dashboard.html`.
- `tests/lint.sh`: linter to run after every change.
- `config.js`: config file containing default server configurations.
- `package.json`: file with nodeJS configurations (dependencies etc...).
- `server.js`: the server which is started when running `npm start`.
