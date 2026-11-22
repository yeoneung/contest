#!/usr/bin/env python3
"""Extract a small, versioned case dataset from public Meta Kaggle CSV files.

No Kaggle credentials are used. Large upstream tables are streamed, never saved.
The retained records contain only competition/team/submission identifiers,
dates, ranks, and scores; participant names are omitted. Pseudonymous leader and author identifiers are
retained solely to establish ownership of a dated source version.
Run from an extracted replication archive: python experiments/fetch_stanford_activity.py
"""
import csv
import argparse
import hashlib
import io
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'stanford_rna'
VERSION = 2401
COMPETITIONS = {'87793', '118765'}
API = 'https://www.kaggle.com/api/v1/datasets/download/kaggle/meta-kaggle/'
csv.field_size_limit(10_000_000)


class HashedStream(io.RawIOBase):
    def __init__(self, source):
        self.source, self.digest, self.size = source, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, buffer):
        block = self.source.read(len(buffer))
        if not block:
            return 0
        self.digest.update(block)
        self.size += len(block)
        buffer[:len(block)] = block
        return len(block)


def extract(name, fields, predicate):
    url = API + name + '?datasetVersionNumber=' + str(VERSION)
    started = time.monotonic()
    request = urllib.request.Request(url, headers={'User-Agent': 'Contest-public-data-replication/1.0'})
    target = OUT / name.lower()
    with urllib.request.urlopen(request, timeout=180) as response:
        headers = {k: response.headers.get(k) for k in ('Content-Length', 'ETag', 'Last-Modified', 'x-goog-generation', 'x-goog-hash')}
        raw = HashedStream(response)
        text = io.TextIOWrapper(io.BufferedReader(raw, buffer_size=1024*1024), encoding='utf-8-sig', newline='')
        reader = csv.DictReader(line.replace('\x00', '') for line in text)
        n = retained = 0
        with target.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                n += 1
                if predicate(row):
                    writer.writerow({key: row.get(key, '') for key in fields})
                    retained += 1
                if n % 2_000_000 == 0:
                    print(name, 'rows', n, 'retained', retained, 'MB', round(raw.size/1e6), flush=True)
    record = dict(url=url, dataset_version=VERSION,
                  retrieved_utc=datetime.now(timezone.utc).isoformat(),
                  headers=headers, source_bytes=raw.size, source_sha256=raw.digest.hexdigest(),
                  source_rows=n, retained_rows=retained,
                  extracted_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  seconds=round(time.monotonic()-started, 2))
    (OUT / (name.lower()+'.provenance.json')).write_text(json.dumps(record, indent=2)+'\n')
    print(name, 'complete:', retained, 'of', n, 'rows', flush=True)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true',help='Download the pinned snapshot again, replacing retained subsets.')
    force=parser.parse_args().refresh
    OUT.mkdir(parents=True, exist_ok=True)
    team_file = OUT / 'teams.csv'
    if force or not (OUT / 'teams.csv.provenance.json').exists():
        extract('Teams.csv', ['Id', 'CompetitionId', 'TeamLeaderId', 'ScoreFirstSubmittedDate',
                'LastSubmissionDate', 'PublicLeaderboardSubmissionId',
                'PrivateLeaderboardSubmissionId', 'IsBenchmark',
                'PublicLeaderboardRank', 'PrivateLeaderboardRank'],
                lambda row: row.get('CompetitionId') in COMPETITIONS)
    with team_file.open() as stream:
        teams = {row['Id'] for row in csv.DictReader(stream)}
    if force or not (OUT / 'submissions.csv.provenance.json').exists():
        extract('Submissions.csv', ['Id', 'TeamId', 'SourceKernelVersionId',
                'SubmissionDate', 'ScoreDate', 'IsAfterDeadline', 'IsSelected',
                'PublicScoreLeaderboardDisplay', 'PublicScoreFullPrecision',
                'PrivateScoreLeaderboardDisplay', 'PrivateScoreFullPrecision'],
                lambda row: row.get('TeamId') in teams)
    if force or not (OUT / 'kernelversions.csv.provenance.json').exists():
        with (OUT / 'submissions.csv').open() as stream:
            versions = {row['SourceKernelVersionId'].encode() for row in csv.DictReader(stream)
                        if row['SourceKernelVersionId']}
        extract_versions(versions)


def extract_versions(identifiers):
    """Scan byte lines quickly; fully parse every selected CSV record.

    Titles can contain embedded newlines. Only selected records are accumulated
    and passed to the strict CSV parser; continuation lines of unselected rows
    cannot match unless they also begin with a requested numeric identifier.
    Schema and numeric/date fields are checked on every retained row.
    """
    name='KernelVersions.csv'
    url=API+name+'?datasetVersionNumber='+str(VERSION)
    fields=['Id','ScriptId','AuthorUserId','ParentScriptVersionId','CreationDate','VersionNumber',
            'IsInternetEnabled','RunningTimeInMilliseconds','AcceleratorTypeId']
    target=OUT/'kernelversions.csv'
    digest=hashlib.sha256();size=lines=retained=0
    started=time.monotonic()
    with urllib.request.urlopen(url,timeout=180) as response, target.open('w',newline='') as output:
        headers={k:response.headers.get(k) for k in ('Content-Length','ETag','Last-Modified','x-goog-generation','x-goog-hash')}
        stream=io.BufferedReader(response,buffer_size=1024*1024)
        header_line=stream.readline();digest.update(header_line);size+=len(header_line)
        columns=next(csv.reader([header_line.decode('utf-8-sig')]))
        writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
        for line in stream:
            digest.update(line);size+=len(line);lines+=1
            end=line.find(b'",',1)
            if line[:1]!=b'"' or end<0 or line[1:end] not in identifiers:
                continue
            text=line.decode('utf8').replace('\x00','')
            while True:
                try:
                    parsed=list(csv.reader(io.StringIO(text),strict=True))
                    if len(parsed)==1 and len(parsed[0])==len(columns):break
                except csv.Error:pass
                more=stream.readline()
                if not more:raise ValueError('Incomplete selected CSV record')
                digest.update(more);size+=len(more);lines+=1
                text+=more.decode('utf8').replace('\x00','')
            row=dict(zip(columns,parsed[0]))
            assert row['Id'].encode() in identifiers and row['ScriptId'].isdigit()
            datetime.strptime(row['CreationDate'],'%m/%d/%Y %H:%M:%S')
            writer.writerow({key:row[key] for key in fields});retained+=1
            if retained%100==0:
                print(name,'matched',retained,'MB',round(size/1e6),flush=True)
    record=dict(url=url,dataset_version=VERSION,retrieved_utc=datetime.now(timezone.utc).isoformat(),
                headers=headers,source_bytes=size,source_sha256=digest.hexdigest(),source_byte_lines=lines,
                requested_version_ids=len(identifiers),retained_rows=retained,
                extracted_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                seconds=round(time.monotonic()-started,2))
    (OUT/'kernelversions.csv.provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(name,'complete',retained,'matches',flush=True)


if __name__ == '__main__':
    main()
