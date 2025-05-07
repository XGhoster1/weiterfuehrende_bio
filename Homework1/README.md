# Homework 01

## Submission
Zip the following modules:

- `exercise1.py`

Name the archive according to the following scheme:

```
f'ex_01_{student.first_name.lower()}_{student.last_name.lower()}_{student.tum_immatriculation_number}.zip'
```
The resulting archive needs to look the following way:
```
ex_01_Gruppe_YOUR-GROUPNUMBER.zip
└── sequence.py
```

Archives with invalid names or invalid structure will not be graded. Since test may be run automatically this leads immediately to a fail.

Submit the archive on Moodle. Make sure that you actually submit the file. Drafts will not be graded.

Pay attention to the deadline on Moodle. Late submissions will not be graded.

## Environment
Most likely it is not necessary but you can create and activate an identical environment with Pyton 3.10 and using the following commands:
```bash
conda env create -f environment.yml
conda activate edu2
```
If you are using Windows, remember to use `Anaconda Prompt`
(refer to https://docs.anaconda.com/anaconda/user-guide/getting-started/#open-prompt-win).



## Testing your code
The following code runs all the tests:

```bash
python -m pytest
```