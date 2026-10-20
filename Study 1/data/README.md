# Input file

The notebook reads one CSV. It is not included here.

| Column | Role |
| --- | --- |
| `title_full` | Title placed in the prompt. `title` is accepted if `title_full` is missing. |
| `abstract` | Abstract placed in the prompt. |
| `stage_2` | Manual full-text label. |
| `stage_3` | Manual final-inclusion label. |
| `ID`, `(internal) id`, or `test_id` | Copied into the result file when present. |

The full run used every row of the file. The Löcherbach file had 3,944 rows. The Birkenmaier file had 917 rows.

`input_template.csv` has the header and no rows.
