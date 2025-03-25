# import torch
# import dataloading
# import model_define

# from model_define import Model
# from pprint import PrettyPrinter

# # Load the trained model
# def Model():
#     best_model = Model()
#     best_model.load_state_dict(torch.load("final_model.pth"))
#     best_model.eval()

#     pp = PrettyPrinter()

# def predict(text: list[str]):
#     embeddings = torch.Tensor(dataloading.embed_text(text))
#     logits = best_model(embeddings)
#     preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
#     scores = torch.softmax(logits, dim=1).detach().cpu().numpy()

#     results = []
#     for t, best_index, score_pair in zip(text, preds, scores):
#         results.append({
#             "text": t,
#             "label": "positive" if best_index == 1 else "negative",
#             "score": score_pair[best_index]
#         })
#     return results


import torch
import dataloading
import model_define
from pprint import PrettyPrinter

def predict_sentiment(texts: list[str]):
    # Load the trained model
    model = model_define.Model()
    model.load_state_dict(torch.load("final_model.pth"))
    model.eval()
    
    # Convert text to embeddings
    embeddings = torch.Tensor(dataloading.embed_text(texts))
    logits = model(embeddings)
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    scores = torch.softmax(logits, dim=1).detach().cpu().numpy()
    
    # Process results
    results = [
        {
            "text": t,
            "label": "positive" if best_index == 1 else "negative",
            "score": score_pair[best_index]
        }
        for t, best_index, score_pair in zip(texts, preds, scores)
    ]
    
    # Print results
    pp = PrettyPrinter()
    pp.pprint(results)
    
    return results

# english_text = "Like any Barnes & Noble, it has a nice comfy cafe, and a large selection of books. The staff is very friendly and helpful. They stock a decent selection, and the prices are pretty reasonable."

# german_translation = "Wie jedes Barnes & Noble hat es ein nettes, gemütliches Café und eine große Auswahl an Büchern. Das Personal ist sehr freundlich und hilfsbereit. Sie haben eine anständige Auswahl und die Preise sind ziemlich vernünftig."

# pp.pprint(predict([english_text, german_translation]))
