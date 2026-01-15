### Introduction
Large Language Models (LLMs) have become increasingly popular in the last few years. Their ability to process natural language and answer using it, simulating a real person, makes them widely used for different tasks. Therefore LLMs are often used as agents - replacing real people and making decisions based on natural language and other forms of data. Thus there is a growing demand in understanding LLMs’ behavior and comparing it to humans. This research focuses on one important aspect of human nature - altruism. It is naturally studied using economic games: Dictator Game and Ultimatum Game. This work applies the same principle by prompting LLMs to simulated economic games. More specifically, dependence of altruism on social distance to the recipient is studied by varying games’ conditions.

### Description of methods
LLMs’ behavior was explored in both dictator games and ultimatum games. Several LLMs were prompted in identical to [20] wording. Similarly to this paper, the social distance was defined as a number from 1 to 100 with 1 indicating the socially closest person, and 100 indicating the 100th closest person to LLM (almost a stranger). Prompted LLMs were chosen as current most popular LLMs as indicated on openrouter.ai website: Google - Gemini 3 Flash Preview, DeepSeek - DeepSeek V3.2, xAI - Grok 4.1 Fast, Anthropic - Claude Sonnet 4.5. The temperature was set at default value 1.

The prompt for the Dictator Game is the following.
You will be asked to make a decision regarding how much
money you might offer another person under a situation
that will be explained shortly. There are no correct or incorrect
answers, and the money is hypothetical – that is, no one will receive
the actual money. Nonetheless, we want you to make your decision
as if the amount and situation were real.
Before describing the situation, we want you to imagine that you
have made a list of the 100 people closest to you in the world,
ranging from your dearest friend or relative at position #1 to a
mere acquaintance at #100. The person at number one would be
someone you know well and is your closest friend or relative. The
person at #100 might be someone you recognize and encounter but
perhaps you may not even know their name. You do not have to
create the list – just imagine that you have done so.
Imagine you have been given the amount of money: $10
You are to divide the amount of money between
yourself and another person who is in place x at that list.
You are free to give as much or as little
of the amount of money as you wish, and you will receive what is
left. Please respond with the amount you wish to offer the other
person. Follow the exact answer template: "I share [decided amount] dollars".

The prompt for the Ultimatum Game was the same as that for the Dictator Game, except for the last paragraph of the instructions and Directions, which read:
Imagine you have been given the amount of money: $10.
You are to divide the amount of money between
yourself and another person who is in place x at that list.
You are free to give as much or as little
of the amount of money as you wish, and you will receive what is
left, but only if the other person accepts your offer. If the other
person rejects your offer, however, then both of you will receive
nothing. Please respond with the amount you wish to offer the other
person. Follow the exact answer template: "I share [decided amount] dollars".

[20] Bechler C, Green L, Myerson J. Proportion offered in the Dictator and Ultimatum Games decreases with amount and social distance. Behav Processes. 2015 Jun;115:149-55. doi: 10.1016/j.beproc.2015.04.003. Epub 2015 Apr 8. PMID: 25862989.

### Code and outputs
The main notebook is `llm_games_openrouter.ipynb`. It prompts OpenRouter
LLM models for Dictator/Ultimatum games, parses responses in the format
"I share [amount] dollars", aggregates results, and saves plots and data
into `out/`.

The script `human_llm_comparison_plots.py` reads the per-model/per-game
CSV outputs from `out/` and combines them with digitized human values to
produce comparison plots.


