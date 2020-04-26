// Copyright (c) Facebook, Inc. and its affiliates.
const path = require('path');
const parse = require('csv-parse/lib/sync');
const express = require('express');
const readLastLines = require('read-last-lines');
const compression = require('compression');
const readdirp = require('readdirp');
const clonedeep = require('lodash.clonedeep');
const AdmZip = require('adm-zip');
const tmp = require('tmp');
const config = require('./config');

app = express();
app.use(compression({filter: () => true}));

/** Filter empty lines or lines with empty fields.
 * Also populate the stats_file field if the runs were loaded recursively.
 */
function processLines(linesStr, recursively, statsFile, dataFile) {
  // Break the string at newlines, to get a list of strings.
  const lines = linesStr.split(/\r?\n/);
  const kept = [];
  for (let line of lines) {
    if (!line.includes('end_status,score,') && line !== '') {
      if (recursively) {
        line += `,${statsFile}`;
      }
      line += `,${dataFile}`,
      kept.push(line);
    }
  }
  return kept;
}

/** Gets a list of available runs, with related information.
 * The files are searched in dataPath (possibily exploring
 * subfolders recursively).
 * readLast is the number of runs to be read starting from
 * the bottom of each stats.csv file. Note that this is not
 * equivalent to the number of runs that will be returned, since
 * lines can be filtered by the processLines function.
 * Returns a list of dicts (JSON).
 */
async function getRunsInfo(dataPath, readLast, recursively) {
  console.log(`Trying to read ${readLast} runs from folder: ${dataPath}.`);

  const header = clonedeep(config.statsHeaders);
  let statsFiles = [];
  if (recursively) {
    const settings = {
      type: 'files',
      fileFilter: [config.data.stats],
    };
    statsFiles = await readdirp.promise(dataPath, settings);
    statsFiles = statsFiles.map((statsFile) => statsFile.fullPath);
    console.log(`Stats files recursively found:\n${statsFiles}`);
    header.push('stats_file');
    header.push('data_file');
  } else {
    statsFiles = [path.join(dataPath, config.data.stats)];
    header.push('data_file');
  }

  // List of strings, each one containing a line.
  let allLines = [];
  for (const statsFile of statsFiles) {
    dataFilePath = path.parse(statsFile);
    dataFile = path.format({
      dir: dataFilePath.dir,
      name: dataFilePath.name,
      ext: '.zip',
    });

    // Read and filter the last lines from each statsFile.
    // The factor 2 is necessary because this library counts every line as two.
    allLines = allLines.concat(await readLastLines.read(statsFile, 2 * readLast)
        .then((lines) => {
          return processLines(lines, recursively, statsFile, dataFile);
        }));
  }

  // Add header to the lines.
  // The header size has to be the same number of fields as the data,
  // otherwise you may get no output from the csv parser.
  allLines.splice(0, 0, header.join(','));

  // Read the data as a csv.
  const runsInfo = parse(allLines.join('\n'), {
    columns: true,
    skip_empty_lines: true,
    comment: '#',
  });
  console.log(`Found ${runsInfo.length} runs.`);
  return runsInfo;
}

/** Creates meaningful error messages with a standard format.
 */
function createErrorMessage(code, request, params='', extraInfo='') {
  return `Call to ${request} returned code ${code}.\n` +
         `=> Parameters:\n${params}\n` +
         `=> Extra info:\n${extraInfo}`;
}

// Serve app.
app.use(express.static(path.join(__dirname + '/app')));
// Serve term.js.
app.use(
    '/third_party/term.js',
    express.static(path.join(__dirname + '/node_modules/term.js/src/term.js')),
);

// Requests.
app.get('/', (req, res) => {
  res.redirect('/dashboard.html');
});
app.get('/runs_info', (req, res) => {
  // Returns a json with the info about the runs.
  // If the path parameter is not set, search the data in
  // config.data.defaultPath.
  // Accepted parameters:
  // - path (string): path to the folder with the data.
  // - readLast (integer): number of lines to read at the bottom of the
  //   stats.csv file(s).
  // - recursively (boolean): if true, search for stats.csv files recursively.
  const dataPath =
    typeof req.query.path !== 'undefined' ? decodeURIComponent(req.query.path) :
                                          config.data.defaultPath;
  const readLast =
    typeof req.query.readLast !== 'undefined' ? parseInt(req.query.readLast) :
                                              config.data.defaultRunsToRead;
  const recursively = req.query.recursively === 'true';

  getRunsInfo(dataPath, readLast, recursively)
      .then((runsInfo) => {
        res.set('Content-Type', 'application/json');
        res.status(200).send(JSON.stringify(runsInfo));
      })
      .catch((error) => {
        if (error.message == 'file does not exist') {
          // Stats file not existent in the specified folder.
          res.status(404).send(
              createErrorMessage(
                  404,
                  '/runs_info',
                  `path: ${req.query.path}`,
                  `No available stats file has been found.\n` +
                  `Path: ${dataPath}.\n` +
                  `Stats file (not found): ` +
                  `${path.join(dataPath, config.data.stats)}.`,
              ),
          );
        } else {
          // Other error.
          res.status(500).send(createErrorMessage(500, '/runs_info'));
          console.log(error);
        }
      });
});
app.get('/ttyrec_file', (req, res) => {
  // Returns the data for a particular ttyrec file, given its absolute
  // filepath.
  // Accepted parameters:
  // - ttyrec: name of the ttyrec file.
  // - datapath: path to the zip file.
  if (typeof req.query.datapath === 'undefined') {
    res.status(400).send(
        createErrorMessage(
            400,
            '/ttyrec_file',
            `ttyrec: ${req.query.ttyrec}`,
            `datapath: ${req.query.datapath}`,
            'No path has been passed to /ttyrec_file.',
        ),
    );
  } else {
    const ttyrecname = decodeURIComponent(req.query.ttyrec);
    let datapath = decodeURIComponent(req.query.datapath);

    if (!path.isAbsolute(datapath)) {
      // Make filepath relative to this folder.
      console.log(`Received a relative datapath: ${datapath}. ` +
                  `Adding this folder as prefix.`);
      datapath = path.join(__dirname, datapath);
    }
    try {
      // need to unzip first into a temp dir
      const datazip = new AdmZip(datapath);
      const name = tmp.tmpNameSync();
      datazip.extractEntryTo(ttyrecname, name + '/');
      const temppath = name + '/' + ttyrecname;
      // send ttyrec over
      res.sendFile(temppath);
      console.log(`Serving ttyrec file: ${temppath}.`);
      // fs.unlinkSync(temppath);
    } catch (error) {
      if (error instanceof TypeError) {
        // File not found.
        res.status(404).send(
            createErrorMessage(
                404,
                '/ttyrec_file',
                `ttyrec: ${ttyrecname}`,
                `datapath: ${datapath}`,
                `File ${temppath} not found.`,
            ),
        );
      } else {
        // Other error.
        res.status(500).send(createErrorMessage(500, '/ttyrec_file'));
        console.log(error);
      }
    }
  }
});

app.listen(config.serverPort).on('error', () => {
  console.log(`ERROR: port ${config.serverPort} already in use.\n` +
              `You may have other servers already running. ` +
              `You can view them by running:\n` +
              `ps aux | grep "node server.js"\n` +
              `(and then you can kill the relevant processes).`);
});
