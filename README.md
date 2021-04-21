# NLP Zara Challenge

A project by Sergi Abashidze, Paula García, Reem Hageali and Sidhant Singhal.

## Abstract

Text summarization is the process of producing concise summaries while preserving significant information. In this project, we investigate the issue of summarizing product titles for Zara given the product description. With the increase in the e-commerce platforms, it is necessary to compress products’ titles for consumer’s convenience. Throughout this paper, we evaluated 5 different models, and the best one turned out to be the  T5ForConditionalGeneration, a state of the art pre-trained generative model. It is an encoder-decoder model and converts all natural language processing problems into text-to-text format. However, the data at hand had a few factors that impacted the model performance, resulting in poor predictions. The product titles included some words that were not present in the description, and there were also some spelling errors in both the product name and description. We were able to tweak the model to gain predictions that were satisfactory after many iterations. 

## 1. Introduction

This project describes the problem that the authors tried to solve when facing the Spain AI Zara Challenge. The task in this hackathon was to generate product names given the product description. This task is interesting because product name is one of the most important factors the consumer considers before making a purchase, so we wanted to make sure that using natural language processing and data-oriented solutions we were able to generate names which will adequately encompass the product’s description. However, after inspecting the data at hand, we became aware of a few factors that were going to impact our performance. First of all, in the training set of the data, we spotted some words present in the product’s name that did not appear in the product’s description. This accounted for 16\% of the tokens, and 13\% of the training observations were affected. Not only this, but the team also spotted some typos in the words conforming the product’s name and description (“with” appeared written as “whit”). Therefore, taking into account that the competition guidelines required an exact match (which included spaces, hyphens, and parentheses) to consider a prediction as a right one, we soon acknowledged that our performance in this hackathon would hardly reach the expected standards. 		 

## 2. Experiments

### 2.1. Data

The competition documentation consists of two datasets. A training dataset, consisting of two columns: product name, and product description; and a testing dataset containing only the product description column. Both of these datasets are in English. Moreover, the training dataset consists of 33613 observations, whereas the testing dataset contains 1441 observations. As explained previously, the task with this data was to use the product description as input, to generate the product name as output. After performing the exploratory data analysis for the training dataset we discovered two main issues. Firstly, there were 16\% of the product name tokens that were not present in the product descriptions, and 13\% of the training observations were affected by this. Secondly, there are some tokens that do not make a lot of sense, as for example: _BLT 05_, _TRSR CMB 05_, or even _BLT BG SRPLS @_. These create big difficulties for us since the predicted product name must be an exact match in order to get points, and to win the competition.

### 2.2. Evaluation method

In order to evaluate our predictions and modelling performance, we are going to rely on Recall-Oriented Understudy for Gisting Evaluation (ROGUE) metric. ROGUE is a set of metrics and a software package used for evaluating automatic summarization and machine translation software in natural language processing. The metric compares an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.

The ROGUE metric consists of:

* __ROGUE - N__, measuring the number of matching ‘n-grams’ between our model-generated text and a ‘reference’
* __ROGUE - L__, measuring the longest common subsequence (LCS) between our model output and reference
* __ROGUE - S__, allowing us to search for consecutive words from the reference text, that appear in the model output but are separated by one-or-more other words.

For each ROGUE, the following is computed:
* __Recall__: counting the number of overlapping n-grams found in both the model output and reference, divided by the total number of n-grams in the reference
* __Precision__: which is calculated the same way as recall but rather than dividing by the reference n-gram count, it is divided by the model n-gram count
* __F-1 Score__: which is the harmonic average of the precision and recall

### 2.3. Experimental details

Throughout the process, we tried five different methods and evaluated which displayed the best performance. We first focused our attention on encoder-decoder models. The first model that was attempted was an encoder-decoder using LSTMs. We tried creating our own neural network, mainly using the TensorFlow library, and with the following layers, in this order: one encoder with embedding layer, three LSTMs, one decoder with embedding layer, one LSTM, one attention layer, and one dense layer. We tried the 'rmsprop', 'Adam' and 'SGD' optimizers, and used the ‘sparse_categorical_crossentropy’ as the loss function. As we saw that this model was not bringing good results, we moved on, and tried a second encoder-decoder, but this time with more layers. For this, we used the torch and TensorFlow libraries, and built our model following this layer architecture: one encoder with embedding layer, one GRU, one decoder with embedding layer, one dropout, one linear, one softmax, one linear, one ReLu, one GRU, one linear, and one softmax. We implement it with the 'SGD' optimizer and 'NLLLoss' as the loss function. Lastly, as this did not work either, we moved to use pre-trained models. Therefore, we continued with the encoder-decoder strategy using Bert. For this, we relied on the torch and transformer libraries and chose 'Adam' as the optimizer.

Unfortunately, these approaches using these encoder-decoders did not work, so we moved on to try different models, and we tried the GPT-2 model from the gpt_2_simple library. To do this, we chose the model size to be 124M, and the optimizer to be 'AdamW'. But it did not work either.

Lastly, we tried our last attempt, and chose another pre-trained model, T5 for Conditional Generation, using the torch and transformers libraries. For this model, we used the 'SGD' optimizer, and finally, were able to get some decent results. 

### 2.4. Results

As for the results, as explained in the section above, we need to say that nearly none of the models were considered to be even performing. Regarding the first model described, the encoder-decoder using LSTMs, this showed no learning at all, as the loss of the model would not decrease throughout the iterations. Something similar happened with our second experiment, an encoder-decoder with more layers, as it did not learn either.

Furthermore, regarding the third experiment, the Bert encoder-decoder, this seemed to learn, but when predicting, it was only able to predict the first token of the sentence, and all of the rest were constant, and not correct. Moving onto the next pre-trained that we tried, we are not able to report results, as we did not have the computational power to evaluate this model, the expected evaluation time consists of 40 hours.

Lastly, regarding our fifth experiment, the pre-trained T5, it was the the only model that we computed the Rouge score for, as it was the only one we could correctly train. The results for this were: 

* ROGUE-1 {f: 0.21, p: 0.23, r:  0.24}
* ROGUE-2 { f: 0.05, p: 0.05, r: 0.06}
* ROGUE-L { f: 0.21 , p: 0.23, r: 0.23}

## 3. Analysis

Previous to analyzing the results, the team thinks that it is important for the reader to notice that for the cases where tokens occurred once or twice in the whole product description collection (e.g. _Disney_), we decided to not include them in the vocabulary, as this was biasing the results.

Generally speaking, the model was able to make predictions; in many instances, it predicted the exact word which had to be predicted just like in the test dataset, yet as for the overall structure of the predicted sentence, the model failed to predict the exact match. Based on the scoring, from the ROGUE metric, we can see that the overlaps of unigrams (ROGUE-1) and bigrams (ROGUE-2) of the predictions were poor compared to the actual results. 

During the modelling, we have come across the problem of overtraining. Moreover, comparing the predicted product name to the actual one, we saw that our model had difficulties predicting long text.

The predictions look like this:

<a href="https://www.linkpicture.com/view.php?img=LPic607f124ecd0eb1799119600"><img src="https://www.linkpicture.com/q/RESULTS.png" type="image"></a>

If you want to take a whole look at them, take a look at the pred_vs_actual.csv in the repository.

## 4. Conclusion

Working with pre-trained models we have come across with two main issues. The first one was that we had to deal with the scarce computational resources that we have. Since the pre-trained models had millions of parameters we had to use Google Colaboratory in order to fit the model which overall took hours for the training process. Sometimes, this was not enough, and so we decided to rely on the cloud services from Azure. Working on this project we have understood how important it is to have enough computational capacity in order to get adequate results and complete this kind of projects. The second issue we came across with was the over-training done by these models which was challenging to deal with.

In addition, even working with state-of-the-art models, our predictions were poor due to the small size and poor structure of the dataset that was provided to us. We saw instances where we had to predict a word that was not even mentioned in the product description column. These types of issues made the project even more difficult. That said, we understood how important it is to have good, structured and valuable dataset since the model will never predict words which it has never learned.

Given this situation that we encountered, we had major difficulties making predictions even using state-of-the-art models like GPT-2, BERT and T5. However, after many iterations and tweakings done to the models, we were able to get predictions that satisfied us. 

Despite poor results (from ROGUE metric), as a team, we are well pleased with our work since this was the first time for us using state-of-the-art methods in Natural Language Processing, and in machine learning overall. The challenges we faced during the completion of this project made us think outside of the box and implement things that we have never done before. 
