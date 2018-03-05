# -*- coding: utf-8 -*-
import csv
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime,timedelta
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import pickle

number_of_posts = 100
print('number of posts ', number_of_posts)

prondict = cmudict.dict()
not_punctuation = lambda w: not (len(w) == 1 and (not w.isalpha()))
get_word_count = lambda text: len(list((filter(not_punctuation, word_tokenize(text)))))
get_sent_count = lambda text: len(sent_tokenize(text))
numsyllables_pronlist = lambda l: len(list(filter(lambda s: int(s.encode('ascii', 'ignore').lower()[-1]), l)))


def numsyllables(word):
    try:
        return list(set(map(numsyllables_pronlist, prondict[word.lower()])))
    except KeyError:
        return [0]


def text_statistics(text):
    word_count = get_word_count(text)
    sent_count = get_sent_count(text)
    syllable_count = sum(map(lambda w: max(numsyllables(w)), word_tokenize(text)))
    return word_count, sent_count, syllable_count


flesch_formula = lambda word_count, sent_count, syllable_count: 206.835 - 1.015 * word_count / sent_count - 84.6 * syllable_count / word_count


def flesch(text):
    word_count, sent_count, syllable_count = text_statistics(text)
    return flesch_formula(word_count, sent_count, syllable_count)


fk_formula = lambda word_count, sent_count, syllable_count: 0.39 * word_count / sent_count + 11.8 * syllable_count / word_count - 15.59


def flesch_kincaid(text):
    word_count, sent_count, syllable_count = text_statistics(text)
    return fk_formula(word_count, sent_count, syllable_count)


class Question:
    all_Questions = []

    def __init__(self, Id, CreationDate, Score, ViewCount, Body, OwnerUserId, Title, Tags, AnswerCount, CommentCount,
                 AcceptedAnswerId):
        self.Id = Id
        self.CreationDate = datetime.strptime(CreationDate, '%Y-%m-%d %H:%M:%S')
        self.Score = Score
        self.ViewCount = ViewCount
        self.Body = Body
        self.OwnerUserId = OwnerUserId
        self.Title = Title
        self.Tags = Tags
        self.AnswerCount = AnswerCount
        self.CommentCount = CommentCount
        self.tag_similarity = 0
        self.Tags = self.Tags.split('<')
        self.answer_class = None
        for i in range(len(self.Tags)):
            self.Tags[i] = self.Tags[i][:-1]
        del self.Tags[0]
        self.question_asked = 0
        self.question_answered = 0
        self.answered = 0
        self.asker_satisfaction = 0
        self.AcceptedAnswerId = AcceptedAnswerId
        self.AcceptedAnswerDuration = -1
        # if AnswerCount and int(AnswerCount) > 0:
        #   self.answered = 1
        if AcceptedAnswerId:
            self.answered = 1

    def set_feature(self):
        self.title_length = len(self.Title)
        self.post_length = len(self.Body)
        # if self.Id != '40004818' and self.Id != '40037764' and self.Id != '7980583' and self.Id != '7980751':
        #     while '<code>' in self.post_length:
        #         index_code = self.post_length.index('<code>')
        #         index_end_code = self.post_length.index('</code>')
        #         self.post_length = self.post_length[:index_code] + self.post_length[index_end_code + 7:]

        self.readability = flesch(self.Body)
            # self.post_length = len(self.post_length)

        tag_similarity = {}
        for tag in self.Tags:
            for q in Question.all_Questions:
                if q != self and q.CreationDate < self.CreationDate and tag in q.Tags:
                    if tag in tag_similarity:
                        tag_similarity[tag] += 1
                    else:
                        tag_similarity[tag] = 1
        if len(tag_similarity.values()) != 0:
            self.tag_similarity = min(tag_similarity.values())
        self.code = 0
        if '<code>' in self.Body:
            self.code = 1

        for q in Question.all_Questions:
            if q != self and q.CreationDate < self.CreationDate and self.OwnerUserId == q.OwnerUserId:
                self.question_asked += 1

        for a in Answer.all_Answers:
            if self.AcceptedAnswerId == a.Id:
                self.AcceptedAnswerDuration = a.CreationDate - self.CreationDate
            if a != self and a.CreationDate < self.CreationDate and self.OwnerUserId == a.OwnerUserId:
                self.question_answered += 1
        self.asker_score = 0
        for u in Karbar.all_Users:
            if u.Id == self.OwnerUserId:
                self.asker_score = u.asker_score
        self.answer_class = 0
        for a in Answer.all_Answers:
            if a.Id == self.AcceptedAnswerId:
                temp = a.CreationDate - self.CreationDate
                time = temp.total_seconds() / 60
                if time <= 1440:
                    self.answer_class = 1
                if 5760 >= time >= 1440:
                    self.answer_class = 2
                elif 5760 < time <= 38880:
                    self.answer_class = 3
                else:
                    self.answer_class = 4

        for u in Karbar.all_Users:
            if u.Id == self.OwnerUserId:
                self.asker_satisfaction = u.satisfaction

    @staticmethod
    def asddQuestions(Question):
        Question.all_Questions.append(Question)


class Answer:
    all_Answers = []

    def __init__(self, Id, CreationDate, OwnerUserId):
        self.Id = Id
        self.CreationDate = datetime.strptime(CreationDate, '%Y-%m-%d %H:%M:%S')
        self.OwnerUserId = OwnerUserId

    @staticmethod
    def addAnswer(Answer):
        Answer.all_Answers.append(Answer)


class Karbar:
    all_Users = []

    def __init__(self, Id, UpVotes, DownVotees):
        self.Id = Id
        self.UpVotes = UpVotes
        self.DownVotes = DownVotees
        self.satisfaction = 0

    def set_feature(self):
        self.asker_score = abs(int(self.UpVotes) - int(self.DownVotes))

        for a in Answer.all_Answers:
            if a.OwnerUserId == self.Id:
                for q in Question.all_Questions:
                    if q.AcceptedAnswerId == a.Id:
                        self.satisfaction += 1

    @staticmethod
    def addUser(User):
        User.all_Users.append(User)


def random_forest(A, v, A_test, v4, label_test, label_test_4class):
    clf = RandomForestClassifier(max_depth=4, random_state=0)
    clf.fit(A, v)
    predicted = clf.predict(A_test)
    data = {}
    data['2class'] = {}
    data['2class']['precision'] = precision_score(label_test, predicted, average='macro')
    data['2class']['recall'] = recall_score(label_test, predicted, average='macro')
    data['2class']['accuracy'] = accuracy_score(label_test, predicted)
    data['2class']['f1'] = f1_score(label_test, predicted, average='macro')
    clf.fit(A, v4)
    predicted = clf.predict(A_test)
    data['4class'] = {}
    data['4class']['precision'] = precision_score(label_test_4class, predicted, average='macro')
    data['4class']['recall'] = recall_score(label_test_4class, predicted, average='macro')
    data['4class']['accuracy'] = accuracy_score(label_test_4class, predicted)
    data['4class']['f1'] = f1_score(label_test_4class, predicted, average='macro')
    with open('random_forest', 'wb') as outfile:
        pickle.dump(data, outfile)


def decision_tree(A, v, A_test, v4, label_test, label_test_4class):
    clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    clf.fit(A, v)
    predicted = clf.predict(A_test)
    data = {}
    data['2class'] = {}
    data['2class']['precision'] = precision_score(label_test, predicted, average='macro')
    data['2class']['recall'] = recall_score(label_test, predicted, average='macro')
    data['2class']['accuracy'] = accuracy_score(label_test, predicted)
    data['2class']['f1'] = f1_score(label_test, predicted, average='macro')
    clf.fit(A, v4)
    predicted = clf.predict(A_test)
    data['4class'] = {}
    data['4class']['precision'] = precision_score(label_test_4class, predicted, average='macro')
    data['4class']['recall'] = recall_score(label_test_4class, predicted, average='macro')
    data['4class']['accuracy'] = accuracy_score(label_test_4class, predicted)
    data['4class']['f1'] = f1_score(label_test_4class, predicted, average='macro')
    favorite_color = {"lion": "yellow", "kitty": "red"}
    with open('decision_tree', 'wb') as outfile:
        pickle.dump(data, outfile)


def svm_linear(A, v, A_test, v4, label_test, label_test_4class):
    clf = svm.LinearSVC(C=1.25)
    clf.fit(A, v)
    predicted = clf.predict(A_test)
    data = {}
    data['2class'] = {}
    data['2class']['precision'] = precision_score(label_test, predicted, average='macro')
    data['2class']['recall'] = recall_score(label_test, predicted, average='macro')
    data['2class']['accuracy'] = accuracy_score(label_test, predicted)
    data['2class']['f1'] = f1_score(label_test, predicted, average='macro')
    clf.fit(A, v4)
    predicted = clf.predict(A_test)
    data['4class'] = {}
    data['4class']['precision'] = precision_score(label_test_4class, predicted, average='macro')
    data['4class']['recall'] = recall_score(label_test_4class, predicted, average='macro')
    data['4class']['accuracy'] = accuracy_score(label_test_4class, predicted)
    data['4class']['f1'] = f1_score(label_test_4class, predicted, average='macro')
    with open('svm_linear', 'wb') as outfile:
        pickle.dump(data, outfile)


def neural_network(A, v, A_test, v4, label_test, label_test_4class):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(A, v)
    predicted = clf.predict(A_test)
    data = {}
    data['2class'] = {}
    data['2class']['precision'] = precision_score(label_test, predicted, average='macro')
    data['2class']['recall'] = recall_score(label_test, predicted, average='macro')
    data['2class']['accuracy'] = accuracy_score(label_test, predicted)
    data['2class']['f1'] = f1_score(label_test, predicted, average='macro')
    clf.fit(A, v4)
    predicted = clf.predict(A_test)
    data['4class'] = {}
    data['4class']['precision'] = precision_score(label_test_4class, predicted, average='macro')
    data['4class']['recall'] = recall_score(label_test_4class, predicted, average='macro')
    data['4class']['accuracy'] = accuracy_score(label_test_4class, predicted)
    data['4class']['f1'] = f1_score(label_test_4class, predicted, average='macro')
    with open('neural_network', 'wb') as outfile:
        pickle.dump(data, outfile)


with open('QueryResults.csv', 'rt') as file:
    reader = csv.reader(file)
    i = 0
    for row in reader:
        i += 1
        if i == number_of_posts + 2:
            break
        if i == 1:
            pass
        else:
            PostTypeId = int(row[1])
            if PostTypeId == 1:
                Question.asddQuestions(
                    Question(row[0], row[4], row[6], row[7], row[8], row[9], row[15], row[16], row[17], row[18],
                             row[2]))
            elif PostTypeId == 2:
                Answer.addAnswer(Answer(row[0], row[4], row[9]))

with open('QueryResultsUser.csv', 'rt') as file:
    reader = csv.reader(file)
    i = 0
    for row in reader:
        i += 1
        if i == 1:
            pass
        else:
            Karbar.addUser(
                Karbar(row[0], row[9], row[10]))
print(len(Question.all_Questions))
print(len(Answer.all_Answers))
for u in Karbar.all_Users:
    u.set_feature()
for q in Question.all_Questions:
    q.set_feature()

matrix_list = []
label = []
label_4class = []
for q in Question.all_Questions[:int(len(Question.all_Questions) * 0.8)]:
    temp_list = []
    temp_list.append(q.title_length)
    temp_list.append(q.post_length)
    temp_list.append(q.tag_similarity)
    temp_list.append(q.code)
    temp_list.append(q.question_answered)
    temp_list.append(q.question_asked)
    temp_list.append(q.asker_score)
    temp_list.append(q.asker_satisfaction)
    matrix_list.append(temp_list)
    label.append(q.answered)
    label_4class.append(q.answer_class)

label_test = []
label_test_4class = []
matrix_list_test = []

for q in Question.all_Questions[int(len(Question.all_Questions) * 0.8):]:
    temp_list = []
    temp_list.append(q.title_length)
    temp_list.append(q.post_length)
    temp_list.append(q.tag_similarity)
    temp_list.append(q.code)
    temp_list.append(q.question_answered)
    temp_list.append(q.question_asked)
    temp_list.append(q.asker_score)
    temp_list.append(q.asker_satisfaction)
    matrix_list_test.append(temp_list)
    label_test.append(q.answered)
    label_test_4class.append(q.answer_class)

A = csr_matrix(matrix_list)
v = np.array(label)
A_test = csr_matrix(matrix_list_test)
v4 = np.array(label_4class)

random_forest(A, v, A_test, v4, label_test, label_test_4class)
decision_tree(A, v, A_test, v4, label_test, label_test_4class)
svm_linear(A, v, A_test, v4, label_test, label_test_4class)
neural_network(A, v, A_test, v4, label_test, label_test_4class)


accepted_answer_durtion_list=[]

for q in Question.all_Questions:
    print(q.AcceptedAnswerDuration)
    if q.AcceptedAnswerDuration != -1:
        accepted_answer_durtion_list.append(q.AcceptedAnswerDuration)

average_timedelta = sum(accepted_answer_durtion_list, timedelta(0)) / len(accepted_answer_durtion_list)

print(average_timedelta)
accepted_answer_durtion_list.sort()
print(accepted_answer_durtion_list[int(len(accepted_answer_durtion_list)/2)])