from transformers import pipeline
import csv

# Load pretrained sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Input number of comments
while True:
    try:
        number_of_comments = int(input('--------------------\n\nEnter how many comments you want for sentiment analysis: '))
        if number_of_comments > 0:
            break
        else:
            print("Please enter a positive integer.")
    except ValueError:
        print("Invalid input. Please enter a number.")

all_comments_result = {'positive': 0, 'negative': 0 }
all_comments = []

# Loop through comments
for i in range(number_of_comments):
    comment = input(f'Enter your comment: (Comment {i+1} out of {number_of_comments}): ')
    comment_result = sentiment_pipeline(comment)[0]  # Analyze sentiment of comment
    # Output
    print(f"\nComment: {comment}")
    print(f"Sentiment: {comment_result['label']}, Confidence: {comment_result['score']:.2f}")
    if comment_result['label'] == 'POSITIVE': all_comments_result['positive'] += 1
    else: all_comments_result['negative']+=1
    print('------------------------')  # Separator for readability

print("All comments result:", all_comments_result)