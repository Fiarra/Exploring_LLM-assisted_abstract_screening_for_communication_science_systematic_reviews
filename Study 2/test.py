#%%
import pandas as pd
from utils.main import classify_one_entry

#%%
res = classify_one_entry(
    project_id="climate",
    entry={"id":"A1","abstract":"We analyze media framing of protests climate change..."},
    system_prompt_path="prompts/system.txt",
    user_prompt_path="prompts/user.txt",
    model="mistral-small",
    decision="ordinal",               
    temperature=0.0,
    topic="climate communication",          
    domain="communication research",    
    codebook_csv="criteria/criteria_test.csv",
    examples_csv="examples/examples_test.csv"
)
# %%
