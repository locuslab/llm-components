import os
import torch
import argparse

from openai import OpenAI


def main(head):
    if not os.path.exists(f'results/llama3.1-8b-it/mean_ablate_100/sentences_{head}.pt'):
        return
    sentences = []
    obj = torch.load(f'results/llama3.1-8b-it/mean_ablate_100/sentences_{head}.pt', weights_only=True)
    # print(head, ':', obj['ablated_acc'])
    sentences_lst = obj['sentences']
    for idx, sample in enumerate(sentences_lst):
        sentences.append(f'Excerpt {idx}: {sample}')
    text = '\n'.join(sentences)

    prompt = '\
    **System:**\
    \nYou are a meticulous AI researcher conducting an important investigation into a specific attention head inside a language model that activates in response to text excerpts from a dataset involving mathematical problem solving. Your overall task is to describe features of text excerpts that cause the attention to strongly activate.\
    \n\nYou will receive a list of text excerpts on which the attention head activates. Tokens causing activation will appear between delimiters like {{this}}. Consecutive activating tokens will also be accordingly delimited {{just like this}}. If no tokens are highlighted with {{}}, then the attention head does not activate on any tokens in the excerpt.\
    \n\nNote: Attention heads activate on a word-by-word basis. Also, attention head activations can only depend on words before the word it activates on, so the description cannot depend on words that come after, and should only depend on words that come before the activation. Note: make your final descriptions as concise as possible, using as few words as possible to describe text features that activate the attention head. Be specific and note that this is a mathematical problem-solving dataset, so be precise as to what words activate within the category of mathematical problem solving. \
    \n\n\n**User:**\
    \nAttention Head 1:\
    \n\nExcerpt 1: Weng {{earns}} $12 an hour for babysitting. Yesterday, she just did {{50}} minutes of babysitting. {{How}} {{much}} did she {{earn}}?\
    \nExcerpt 2: Betty is saving {{money}} for a new wallet which costs ${{100}}. Betty has only {{half}} of the {{money}} she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. {{How}} much more money does Betty need to buy the wallet?\
    \nExcerpt 3: Ashley has 10 {{dollars}} and needs to {{purchase}} a toy that will {{cost}} 3 {{dollars}}; how much {{money}} will she have after she pays?\
    \nExcerpt 4: Ben found 50 {{cents}} in his pocket and wants to {{spend}} his {{money}} to {{buy}} a candy bar that {{costs}} 30 cents; how much change will he have?\
    \nExcerpt 5: Carol uses 4 {{coins}} to {{pay}} for an apple that has a {{price}} of 1 {{dollar}}; how many {{coins}} does she have left?\
    \nExcerpt 6: Dan has 8 {{dollars}} and wants to {{save}} half of his {{money}} for a gift; how many {{dollars}} will he have left to {{spend}}?\
    \nExcerpt 7: Jill {{earns}} 5 {{dollars}} per hour doing chores and works 3 hours; how much {{money}} does she have, how much does she {{save}} if she puts half away, and how much can she {{spend}}?\
    \nExcerpt 8: Mark receives a weekly {{wage}} of 7 {{dollars}} for mowing lawns; after 4 weeks, how much {{income}} has he made, and if he {{saves}} 10 dollars, how much is left to {{spend}}?\
    \nExcerpt 9: Emma wants to {{budget}} her 20 {{dollars}} so she can {{save}} for a new game; if she sets aside 5 dollars each week, how many weeks until she reaches her {{goal}} of 20 dollars, and how much {{money}} remains for snacks?\
    \nExcerpt 10: Noah needs to pay 5 dollars for a trip and {{earns}} 2 {{dollars}} daily from pet-sitting; how many days until he has enough {{money}}, and if he keeps earning for 3 more days, how much does he {{save}} if he puts half away, and how much can he {{spend}}?\
    \n\n**Assistant:**\
    \n[DESCRIPTION:] phrases that refer to money\
    \n\n\n**User:**\
    \nAttention Head 2:\
    \n\nExcerpt 1: Weng earns $12 an hour for babysitting. Yesterday, {{she}} just did {{50}} minutes of babysitting. {{How}} {{much}} did {{she}} earn?\
    \nExcerpt 2: Betty is saving money for a new wallet which costs $100. Betty has only half of the money {{she}} needs. {{Her}} parents decided to give {{her}} $15 for that purpose, and {{her}} grandparents twice as much as {{her}} parents. {{How}} much more money does Betty need to buy the wallet?\
    \nExcerpt 3: Ashley has 10 dollars and needs to purchase a toy that will cost 3 dollars; how much money will {{she}} have after {{she}} pays?\
    \nExcerpt 4: Ben found 50 cents in {{his}} pocket and wants to spend {{his}} money to buy a candy bar that costs 30 cents; how much change will {{he}} have?\
    \nExcerpt 5: Carol uses 4 coins to pay for an apple that has a price of 1 dollar; how many coins does {{she}} have left?\
    \nExcerpt 6: Dan has 8 dollars and wants to save half of {{his}} money for a gift; how many dollars will {{he}} have left to spend?\
    \nExcerpt 7: Jill earns 5 dollars per hour doing chores and works 3 hours; how much money does {{she}} have, how much does {{she}} save if {{she}} puts half away, and how much can {{she}} spend?\
    \nExcerpt 8: Mark receives a weekly wage of 7 dollars for mowing lawns; after 4 weeks, how much income has {{he}} made, and if {{he}} saves 10 dollars, how much is left to spend?\
    \nExcerpt 9: Emma wants to budget {{her}} 20 dollars so {{she}} can save for a new game; if {{she}} sets aside 5 dollars each week, how many weeks until {{she}} reaches {{her}} goal of 20 dollars, and how much money remains for snacks?\
    \nExcerpt 10: Noah needs to pay 5 dollars for a trip and earns 2 dollars daily from pet-sitting; how many days until {{he}} has enough money, and if {{he}} keeps earning for 3 more days, how much does {{he}} save if {{he}} puts half away, and how much can {{he}} spend?\
    \n\n**Assistant:**\
    \n[DESCRIPTION:] pronouns\
    \n\n\n**User:**\
    \nAttention Head 3:\
    \n\nExcerpt 1: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\
    \nExcerpt 2: Betty is saving money for a new wallet which costs $100. Betty has only {{half}} of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents {{twice}} as much as her parents. How much more money does Betty need to buy the wallet?\
    \nExcerpt 3: Ashley has 10 dollars and needs to purchase a toy that will cost 3 dollars; how much money will she have after she pays?\
    \nExcerpt 4: Ben found 50 cents in his pocket and wants to spend his money to buy a candy bar that costs 30 cents; how much change will he have?\
    \nExcerpt 5: Eva spends {{one-quarter}} of her savings on a new puzzle. How much does she have left if she originally saved $12?\
    \nExcerpt 6: Dan has 8 dollars and wants to save {{half}} of his money for a gift; how many dollars will he have left to spend?\
    \nExcerpt 7: Jill earns 5 dollars per hour doing chores and works 3 hours; how much money does she have, how much does she save if she puts {{half}} away, and how much can she spend?\
    \nExcerpt 8: Mark receives a weekly wage of 7 dollars for mowing lawns; after 4 weeks, how much income has he made, and if he saves 10 dollars, how much is left to spend?\
    \nExcerpt 9: Hugo needs to donate {{one-third}} of his earnings to a charity box. If he earned $9, how much does he donate?\
    \nExcerpt 10: Noah needs to pay 5 dollars for a trip and earns 2 dollars daily from pet-sitting; how many days until he has enough money, and if he keeps earning for 3 more days, how much does he save if he puts {{half}} away, and how much can he spend?\
    \n\n**Assistant:**\
    \n[DESCRIPTION:] fraction or ratio references\
    \n\n**User:**\
    \nAttention Head 4: \n\n' + \
    text + \
    '\n\n**Assistant:** \
    \n[DESCRIPTION:]'


    client = OpenAI(
        api_key='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',  # This is the default and can be omitted
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",
    )
    print(head, ' : ', text)
    print(head, ':', chat_completion.choices[0].message.content)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=str)
    args = parser.parse_args()
    main(args.head)

    # for l in range(32):
    #     for h in range(32):
    #         head = f'L{l}H{h}'
    #         main(head)