
from datetime import timedelta
import datetime


class Event:

    def __init__(self, game_json, eventID):
        self.event_json = game_json['liveData']['plays']['allPlays'][eventID]
        self.event_id = eventID
        self.game_json = game_json

        try:
            self.x_coordinate = self.event_json['coordinates']['x']
        except:
            self.x_coordinate = None
        try:
            self.y_coordinate = self.event_json['coordinates']['y']
        except:
            self.y_coordinate = None

        try:
            self.team = self.event_json['team']['id']
        except:
            self.team = None


        ts = '1900-01-01 00:'
        f = '%Y-%m-%d %H:%M:%S'
        periodTime = datetime.datetime.strptime(ts + str(self.event_json['about']['periodTime']), f) + timedelta(minutes=(20 * (int(self.event_json['about']['period']) - 1)))
        self.period_time = str(periodTime)
        self.period = self.event_json['about']['period']
        self.event_type = self.get_event_type()
        self.event_description = self.get_event_description()
        self.playerid = self.get_player()
        self.secondaryPlayerID = self.get_secondary_player()
        self.penaltySeverity = self.get_penalty_severity()
        self.penaltyMinutes = self.get_penalty_minutes()
        self.state = "5v5"



    def get_player(self):
        playerID = None
        try:
            playerID = self.event_json['players'][0]['player']['id']
            return playerID
        except:
            return playerID

    def get_secondary_player(self):
        playerID = None
        try:
            playerID = self.event_json['players'][1]['player']['id']
            return playerID
        except:
            return playerID


    def get_event_type(self):
        type = ""
        try:
            type = self.event_json['result']['eventTypeId']
        except:
            type = "UNK"
        return type

    def get_event_description(self):
        type = ""
        try:
            type = self.event_json['result']['description']
        except:
            type = ""
        return type

    def get_penalty_severity(self):
        severity = ""
        try:
            severity = self.event_json['result']['penaltySeverity']
        except:
            severity = ""
        return severity

    def get_penalty_minutes(self):
        minutes = 0
        try:
            minutes = self.event_json['result']['penaltyMinutes']
        except:
            minutes = 0
        return minutes
