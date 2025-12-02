# simple calorie lookup mapping
CAL_DB = {
"pizza": 266,
"burger": 295,
"salad": 33,
"pasta": 131,
"taco": 226,
"sushi": 130,
"fried_rice": 163
}




def get_calories(food_label):
    key = food_label.lower().replace(" ", "_")
    return CAL_DB.get(key, 250) # default estimate