from datetime import timedelta
from datetime import datetime
import math

class Shot:

    def __init__(self, game_json, eventID):
        self.shot_base_types = ['Wrist Shot', 'Missed', 'Slap Shot', 'Snap Shot', 'Backhand', 'Blocked', 'Tip-In', 'Deflected', 'Wrap-around', 'Fighting', 'Error']
        self.all_base_types = ['Wrist Shot', 'Missed', 'Slap Shot', 'Snap Shot', 'Backhand', 'Blocked', 'Tip-In', 'Deflected', 'Wrap-around',
                               'Holding', 'PS - Slash on breakaway', 'Hooking', 'Tripping', 'Unsportsmanlike conduct',
                               'PS - Hooking on breakaway', 'Fighting', 'Error', 'Diving']
        self.event_json = game_json['liveData']['plays']['allPlays'][eventID]
        self.event_id = eventID
        self.game_json = game_json

        # Make sure previous event is in the same period
        if eventID - 1 >= 0 and (game_json['liveData']['plays']['allPlays'][eventID - 1]['about']['period'] == self.event_json['about']['period']):
            self.prev_event_json = game_json['liveData']['plays']['allPlays'][eventID - 1]
        else:
            self.prev_event_json = None
        try:
            self.x_coordinate = self.event_json['coordinates']['x']
        except:
            self.x_coordinate = None
        try:
            self.y_coordinate = self.event_json['coordinates']['y']
        except:
            self.y_coordinate = None

        self.x_coordinate_adj = None
        self.y_coordinate_adj = None
        if self.x_coordinate is not None and self.y_coordinate is not None and self.x_coordinate < 0:
            self.x_coordinate_adj = self.x_coordinate * -1
            self.y_coordinate_adj = self.y_coordinate * -1

        try:
            self.team = self.event_json['team']['id']
        except:
            self.team = 0

        ts = '1900-01-01 00:'
        f = '%Y-%m-%d %H:%M:%S'
        periodTime = datetime.strptime(ts + str(self.event_json['about']['periodTime']), f) + timedelta(minutes=(20 * (int(self.event_json['about']['period']) - 1)))
        self.period_time = periodTime
        self.period = self.event_json['about']['period']

        try:
            self.periodtype = self.event_json['about']['periodType']
        except:
            self.periodtype = None

        self.shot_type = self.get_event_type()
        self.shot_distance = self.get_shot_distance()
        self.e_w_diff_last_event = self.get_e_w_diff_last_event()
        self.n_s_diff_last_event = self.get_n_s_diff_last_event()
        self.success = self.get_scored()
        self.shot_angle = self.get_shot_angle()
        self.last_event_type = self.get_last_event_type()
        self.time_since_last_event = self.get_time_since_last_event()
        self.previous_event_distance = self.get_previous_event_distance()
        self.game_time_seconds = self.get_game_time_seconds()
        self.prev_event_was_friendly = self.get_previous_team()


    def get_shot_distance(self):
        x = self.x_coordinate_adj if self.x_coordinate_adj is not None else self.x_coordinate
        y = self.y_coordinate_adj if self.y_coordinate_adj is not None else self.y_coordinate

        try:
            dist = math.sqrt(((87.95 - abs(x)) ** 2) + (y ** 2))
        except:
            return ""
        return round(dist, 2)

    def get_previous_event_distance(self):
        if self.prev_event_json is None:
            return ""

        try:
            dist = round(math.sqrt(((self.prev_event_json['coordinates']['x'] - self.x_coordinate) ** 2) + ((self.prev_event_json['coordinates']['y'] - self.y_coordinate) ** 2)),2)
        except:
            return ""

        return round(dist, 2)

    def get_shot_angle(self):
        try:
            x = self.x_coordinate_adj if self.x_coordinate_adj is not None else self.x_coordinate
            y = self.y_coordinate_adj if self.y_coordinate_adj is not None else self.y_coordinate
            delta_x = (87.95 - abs(x)) ** 2
            delta_y = (y) ** 2
            theta_radians = math.asin(y / (math.sqrt(delta_x + delta_y)))
            theta_radians = (theta_radians * 180) / 3.14
        except:
            return ""
        return round(theta_radians, 2)

    def get_time_since_last_event(self):
        if self.prev_event_json is None:
            return None

        ts = '1900-01-01 00:'
        f = '%Y-%m-%d %H:%M:%S'

        periodAddition = timedelta(minutes=20) * (int(self.prev_event_json['about']['period']) - 1)
        periodTime = datetime.strptime(ts + str(self.prev_event_json['about']['periodTime']), f) + periodAddition
        tdelta = self.period_time - periodTime
        return tdelta.total_seconds()

    def get_e_w_diff_last_event(self):
        if self.prev_event_json is None:
            return ""

        try:
            diff = abs(self.x_coordinate - self.prev_event_json['coordinates']['x'])
        except:
            diff = ""
        return diff

    def get_n_s_diff_last_event(self):
        if self.prev_event_json is None:
            return ""

        try:
            diff = abs(self.y_coordinate - self.prev_event_json['coordinates']['y'])
        except:
            diff = ""
        return diff

    def get_event_type(self):
        try:
            return self.event_json['result']['secondaryType']
        except:
            return "Missed"

    def get_last_event_type(self):
        if self.prev_event_json is None:
            return "Unknown"

        try:
            return self.prev_event_json['result']['eventTypeId']
        except:
            return "Unknown"

    def get_scored(self):
        if self.event_json['result']['eventTypeId'] == "GOAL":
            return 1
        return 0

    def get_game_time_seconds(self):
        ts = '1900-01-01 00:00:00'
        f = '%Y-%m-%d %H:%M:%S'
        return (self.period_time - datetime.strptime(ts, f)).total_seconds()

    def get_previous_team(self):
        try:
            enemy_team = self.prev_event_json['team']['id']
            return 1 if self.team == enemy_team else 0
        except:
            return 0

