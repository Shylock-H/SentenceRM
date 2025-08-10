import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url, data, score_keys=["rewards", "sentence_rewards"], try_max_times=10):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            if isinstance(score_keys, str):
                score_keys = [score_keys]
            results = []
            for key in score_keys:
                assert key in response, f"{key} not in {response}"
                results.append(response.get(key))
            return tuple(results) if len(results) > 1 else results[0]
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(api_url, raw_prompts, answers, score_keys=["rewards", "sentence_rewards"]):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    raw_prompts: original prompts (optional)
    answers: answers with sentence splitting marks (optional)
    score_keys: List of keys for reward scores to extract from response
    """
    # 准备请求数据
    data = {
        "raw_prompts": raw_prompts, 
        "answers": answers
    }

    results = request_api_wrapper(api_url, data, score_keys)
    
    if isinstance(results, tuple):
        rewards, sentence_rewards = results
        return sentence_rewards
    else:
        return torch.tensor(results)


@ray.remote
def remote_rm_fn_ray(api_url, raw_prompts, answers, score_keys=["rewards", "sentence_rewards"]):
    return remote_rm_fn(api_url, raw_prompts, answers, score_keys)


if __name__ == "__main__":
    # test utils
    url = "http://localhost:5000/get_reward"

    answer = "<|user|>\nCoors is in the process of going through a thorough supplier check list to slim down\non the number of suppliers. Coors wants to find out if whether or not its suppliers\ncan increase the capacity and whether or not the suppliers are willing to give Coors\nmore or less free access to the supplier’s production apparatus. Today the meeting\nis with one of the industrial farmers that have one of the largest farms producing\namong other crops barley, wheat, oat, rye and corn for numerous major customers.\nYou are the farmer that Coors is coming to visit today. You know that your farm has\na very attractive size and that it would be possible for you to even grow more on\nyour land. You have even had your eyes on the connecting fields to the far east of\nyour land. You have been contacting the farmer that currently owns that piece of\nland. You are not planning to let Coors know this as you do not want competition for\nthe land. But you will air it if it is beneficial to you to be able to increase your\nproduction to the desired level, only you will not tell Coors that it will not be on\nexisting land.\nYour plan is to keep complete independence and to maintain your current\ncustomers. You have no desire to let yourself become vulnerable by only supplying\nto one customer. Even though Coors has been one of the more stable customers, it\nis still unsound business to rely only on one source. Everybody though has a price, so\nif the price is high enough then you have may even go in a totally different direction.\nBut your main interest is to keep things as they are and still maximise the deal with\nCoors so that you can provide them with the amount of barley and wheat that Coors\nmay want.\nAs for your asking price you will naturally try to take as much as you possible can\nget. The normal price for per kilo of barley is $2 herein included freight to nearest\ntrain station; wheat rates between $3-5 per kilo. The rates are, however, subject to\nordinary supply and demand and naturally also subject to level of quality. Your\nbarley and wheat are at the top of the quality latter. Your current production\ncapacity is 200 tonnes annually of the barley and 100 tonnes of the wheat. Currently\nyou are selling 40 per cent of your barley and wheat to Anheuser-Busch and the\nremainder to Coors. You realise<END> that in order to make the best deal,<END> that the Coors\nrepresentative would not be a decision making person but will probably be an\nintermediate person that will inform the company, and that this representative may\nnot fully grasp the complete complexity of the supply/demand situation even if<END> it is\nhis job.\n<END>You arrange an option for the Coors representative to see your farm's facilities.\n<END>The aim of the visit will be decided by the Coors representative but you will make\nsure to include a presentation of both current production capabilities and projected\ngrowth.<END> You also understand that in the mean time your farmer colleague to the far\neast who already has been contacted, he may have prizes that may interest Coors.\n<END>But Coors welcomes advanced planning, as long as profitability is not affected, \nthen the longer the better, in their opinion.<END> You lay the table in a way that it\ndelays any decision regarding the purchase of your crops until after December.\n<END>When you meet with the Coors representative make sure you:\n<END>1.  Discuss the crop production with the Coors representtive\n<END>2.  Let the representative know that it will be possible to increase the\nproduction of both barley and wheat, by 100 per cent without any\nproblems because of new lands and better irrigation and fertilization.\n<END>3.  Stress that your production this year was operational at full capacity.\n<END>4.  In order to get an impression of demand, ask the Coors\nrepresentative about upcoming production plan and how much more\nCoors will need next year. \n<END>5.  Make sure the Coors representative know you are particular on product\nquality and you have the capacity for quality checks both currently and\nin the future with the predicted production increase.\n<END>6. Officially drop hints about the other customer that could supply Coors. \n<END>7.  Set up the negotiations so that the timeline of any purchase order is delayed until\nafter December.\n<END>a.  Convey to the Coors rep.<END> that a delay will be very difficult\n<END>b. <END> Suggest an official written agreement that guarantees the minimum\npurchase amount,<END> quantity and price.\n<END>c.  Suggest using the Coors rep.<END> as inform\n<END>8. <END> When the Coors representative inquires about planting season and\nyou freeze on the question what do you say?\n<END>9.  Realise that the Coors rep. may suggest working with other farms\nclose to a port, and that this may disturb the extending agreement\nprocurement strategy, how would you counter<|end_of_text|><END>"
    prompt = "Human: If I go to a movie, is there a way to see it for free? Assistant: "
    answer = "Seeing a movie for free without taking any illegal actions, like watching pirated content, can be challenging.<END> However, there are some legitimate ways to potentially see a movie for free or at a very low cost:\n\n<END>1. **Go to Film Festivals:<END>**<END> Many film festivals around the world provide free access to their screenings, especially for certain categories or for filmmakers. You might be able to get a pass or individual tickets for free, depending on the festival's policy.\n\n<END>2. **Film Societies and Clubs:<END>**<END> Some film societies or clubs may offer free screenings as part of their series. It's worth checking with your local community center or college if they run such groups.\n\n<END>3. **Online Contest Winners:<END>** Some organizations host contests where the winners get free movie tickets or passes to see unreleased movies. Keep an eye out for these.\n\n<END>4. **Cinemania or Blockbuster Clubs:<END>** Some communities have book, video, or CD clubs where people can get membership discounts or even free items over time.<END> Occasionally, these clubs offer free movie tickets as part of their services.\n\n<END>5. **Streaming and Entertainment Platforms:<END>**<END> If there are streaming services or entertainment platforms available in your area that offer free trials, you might be able to watch a movie for free during the trial period.<END> However, this doesn't apply for all movies, especially for the newest releases.\n\n<END>6. **Library Events:<END>**<END> Some libraries host movie screenings as part of their programming. These are usually free and might include films that have been released recently.\n\n<END>7. **Partnerships with Companies:<END>**<END> Sometimes, there are partnerships between movie theaters and other businesses that allow you to get a free movie ticket through a promotional offer.<END> Keep an eye on local deals in the area or on company loyalty programs.\n\n<END>Remember, while it's great to be thrifty and try to get the most out of your entertainment dollar, it's also important to respect copyright laws and the efforts of filmmakers and producers who make movies.<END> Illegal streaming and downloading of movies can harm those industries and is considered piracy, which is both illegal and unethical.<END>"
    score = remote_rm_fn(url, [prompt], [answer])
    print(len(score[0]))
