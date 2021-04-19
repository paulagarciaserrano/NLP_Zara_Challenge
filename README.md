# NLP Zara Challenge

## 1. Introduction

This project describes the problem that the authors tried to solve when facing the Spain AI Zara Challenge. The task in this hackathon was to generate product names given the product description. This task is interesting because product name is one of the most important factors the consumer considers before making a purchase, so we wanted to make sure that using natural language processing and data-oriented solutions we were able to generate names which will adequately englobe the product’s description. However, after inspecting the data at hand, we became aware of a few factors that were going to impact our performance. First of all, in the training set of the data, we spotted some words present in the product’s name that did not appear in the product’s description. This accounted for 16% of the tokens, and 13% of the training observations were affected. Not only this, but the team also spotted some typos in the words conforming the product’s name and description (“with” appeared written as “whit”). Therefore, taking into account that the competition guidelines required an exact match (which included spaces, hyphens, and parentheses) to consider a prediction as a right one, we soon acknowledged that our performance in this hackathon would reach the expected standards. 	
