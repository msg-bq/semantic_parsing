def _concept_instance_prompt(concept: str) -> str:
    instruct = '''You are an instantiation generator, when l give you a concept like "date time", "common weather adjective", "weather attribute", etc. you need to give some detailed example instances about it.
The "common weather adjective" represent want to get common weather conditions adjective such as rainy, snowy, sunny and so on. "date time" indicates the time that want to get the weather, which can be vague like "tonight, tomorrow" or precise like "this Monday, next Friday, March 5" and so on.
Don't give the process or any extra words, just return 10 various instances. Again, no extra words and no extra punctuations.
    
######### examples ###########
Question: Please randomly give an instance about "date time"
Answer: ["Tonight", "today", "15 May", "May 2024", "2020.3.15", ...]
    
Question: Please randomly give an instance about "location"
Answer: ["sydney", "China", ...]
  
Question: Please randomly give an instance about "common weather adjective"
Answer: ["snowy", "sunny", "cloudy",  ...]
    
Question: Please randomly give an instance about "{}"\nAnswer:'''

    return instruct.format(concept)