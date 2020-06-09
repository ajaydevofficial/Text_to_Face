import os, sys
sys.path.append(os.path.dirname(os.path.abspath('extractor.py')))
from django.db import connection


def pro_gan(caption):
    with connection.cursor() as cursor:
        words = ['the', 'a', 'an', 'with', 'where', 'in', 'on', 'at', 'of', 'by']
        caption_splitted = caption.split(' ')
        items = []
        gender = None
        race = None
        age = None
        hair = None
        physic = None
        for each in caption_splitted:
            if each not in words:
                word = each

                if word == 'male' or word == 'man' or word == 'men' or word == 'boy':
                    gender = 'male'
                    continue
                elif word == 'female' or word == 'woman' or word == 'lady' or word == 'girl' or word == 'women':
                    gender = 'female'
                    continue

                if word == 'white' or word == 'bright' or word == 'american' or word == 'european' or word == 'light' or word == 'fair':
                    race = 'white'
                    continue
                elif word == 'black' or word == 'dark' or word == 'dull' or word == 'african':
                    race = 'black'
                    continue
                elif word == 'brown' or word == 'asian' or word == 'shady':
                    race = 'brown'
                    continue

                if word == 'young' or word == 'small' or word == 'teen' or word == 'teenage' or word == 'teenager':
                    age = "young"
                    continue
                elif word == 'old' or word == 'mature' or word == 'elder' or word == 'elderly' or word == 'senior':
                    age = 'old'
                    continue
                elif word == 'middle' or word == 'middleage':
                    age = 'middle'
                    continue

                if word == 'fat' or word == 'chubby':
                    physic = 'fat'
                    continue
                elif word == 'slim' or word == 'skinny':
                    physic = 'slim'
                    continue
                elif word != 'blonde' and word !='bald':
                    physic = 'normal'
                    continue

                if word == 'blonde':
                    hair = 'blonde'
                    continue
                elif word == 'bald':
                    hair = 'bald'
                    continue
                else:
                    hair = 'normal'
                    continue
        normal_query = "SELECT image from Image WHERE"
        query = "SELECT image from Image WHERE"
        if gender:
            if query == normal_query:
                query = query + " gender=" + "'" + gender + "'"
            else:
                query = query + " AND gender=" + "'" + gender + "'"
        if race:
            if query == normal_query:
                query = query + " race=" + "'" + race + "'"
            else:
                query = query + " AND race=" + "'" + race + "'"
        if age:
            if query == normal_query:
                query = query + " age=" + "'" + age + "'"
            else:
                query = query + " AND age=" + "'" + age + "'"
        if hair:
            if query == normal_query:
                query = query + " hair=" + "'" + hair + "'"
            else:
                query = query + " AND hair=" + "'" + hair + "'"
        if physic:
            if query == normal_query:
                query = query + " physic=" + "'" + physic + "'"
            else:
                query = query + " AND physic=" + "'" + physic + "'"
        try:
            response = []
            value = list(cursor.execute(query))
            for ob in value:
                response.append(ob[0])
            return(response)
        except Exception as e:
            print("Error Occured")
