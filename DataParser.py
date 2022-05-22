import mysql.connector
import json

from Event import Event
from Shot import Shot
import datetime
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from datetime import date
from datetime import timedelta
import MySQLdb
import MySQLdb.cursors

mydb = MySQLdb.connect(
        host="localhost", user="root", passwd="root", db="nhl",
        cursorclass=MySQLdb.cursors.SSCursor) # put the cursorclass here
mycursor = mydb.cursor()

#mydb = mysql.connector.connect(
#    host="localhost",
#    user="root",
#    password="root",
#    database="nhl",
#    cursorclass = MySQLdb.cursors.SSCursor
#)

#mycursor = mydb.cursor()

"""
Purpose: Get JSON for all games in a regular season
Returns: Array of json for each game
"""
def get_regular_season(year):
    games = []
    gameID = 1

    stringGameID = str(gameID).zfill(4)
    response = requests.get(
        "https://statsapi.web.nhl.com/api/v1/game/" + str(year) + "02" + stringGameID + "/feed/live")
    data = response.json()

    while "messageNumber" not in data:
        if data['gameData']['status']['abstractGameState'] == "Final":
            games.append(data)
        gameID += 1
        stringGameID = str(gameID).zfill(4)
        # response = requests.get("https://statsapi.web.nhl.com/api/v1/game/" + str(year) + "02" + stringGameID + "/feed/live")

        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        response = session.get(
            "https://statsapi.web.nhl.com/api/v1/game/" + str(year) + "02" + stringGameID + "/feed/live")

        data = response.json()

    return games


"""
Purpose: Parse scoring plays from a games JSON
Returns: Scoring play eventIDs  
"""
def parse_scoring_plays(game):
    return game['liveData']['plays']['scoringPlays']


"""
Purpose: Get event id's of all shots in a game
Returns: List of event id's for shots in a game
"""
def parse_plays_single_game(game):
    shots = list()
    events = list()
    eventid = 0
    for play in game['liveData']['plays']['allPlays']:
        shot_time = play['about']['periodTime']
        shot_period = play['about']['period']
        event_type = play['result']['eventTypeId']

        try:
            if (event_type == 'SHOT' or event_type == 'MISSED_SHOT'):  # removed blocked shots
                shots.append(eventid)
            elif (event_type == 'GOAL'):
                shots.append(eventid)
                events.append(eventid)
            else:
                events.append(eventid)
                eventid = eventid + 1
                continue
        except IndexError:
            eventid = eventid + 1
            continue
        eventid = eventid + 1
    return shots, events


"""
Purpose: Parse shot data from a given game
Returns: Dictionary of shot data
"""


def build_shot_data_single_games(game, shots, shot_data=None):
    if shot_data is None:
        shot_data = []
    counter = 0
    for shot in shots:
        # create new shot event
        # it appears that eventIDx can be wrong sometimes. Need to wrap this call in a try-catch
        new_shot = Shot(game, shot)
        try:
            shooter = game['liveData']['plays']['allPlays'][shot]['players'][0]['player']['fullName']
        except:
            counter = counter + 1
            continue
        if counter != 0 and new_shot.prev_event_json is None:
            counter = counter + 1
            continue

        shot_data.append(new_shot)

        # increment counter
        counter = counter + 1

    return shot_data


def build_event_data_single_game(game, events):
    event_data = list()
    counter = 0
    for event in events:
        # create event
        # it appears that eventIDx can be wrong sometimes. Need to wrap this call in a try-catch
        try:
            new_event = Event(game, event)
        except:
            counter = counter + 1
            continue

        event_data.append(new_event)

        # increment counter
        counter = counter + 1

    return event_data


def parse_goals(shot, gameID):
    aOne = 0
    aTwo = 0
    goalie = 0

    playerids = [x['player']['id'] for x in shot.event_json['players'] if x['playerType'] == 'Assist']
    if playerids is not None:
        if len(playerids) == 2:
            aOne = playerids[0]
            aTwo = playerids[1]
        elif len(playerids) == 1:
            aOne = playerids[0]
    goalieid = [x['player']['id'] for x in shot.event_json['players'] if x['playerType'] == 'Goalie']
    if goalieid is not None and len(goalieid) > 0:
        goalie = goalieid[0]

    return (
    int(gameID), str(shot.period_time), int(shot.period), int(shot.event_json['players'][0]['player']['id']), int(aOne),
    int(aTwo), int(goalie))


def extract_shot_data_from_game(shots, gameID):
    arr = []
    goals = []
    for shot in shots:
        a = datetime.datetime(1900, 1, 1, 00, 00, 00)

        ts = '1900-01-01 00:'
        tsv = '1900-01-01 0'
        f = '%Y-%m-%d %H:%M:%S'

        if shot.success == 1 and shot.periodtype != 'SHOOTOUT':
            goals.append(parse_goals(shot, gameID))

        arr.append((int(shot.event_json['players'][0]['player']['id']), int(gameID), int(shot.event_json['team']['id']),
                    str(shot.shot_type), int(shot.period), str(shot.period_time), shot.shot_distance,
                    shot.shot_angle, shot.x_coordinate, shot.y_coordinate,
                    shot.e_w_diff_last_event, shot.n_s_diff_last_event,
                    # time since last shift change
                    0,
                    # game time seconds
                    int(shot.game_time_seconds),
                    # state
                    "",
                    json.dumps(shot.prev_event_json),
                    str(shot.last_event_type),
                    str(shot.time_since_last_event),
                    shot.previous_event_distance,
                    # previous event was friendly
                    shot.prev_event_was_friendly,
                    # shots last ten
                    0,
                    # score
                    0,
                    # is net empty
                    0,
                    # off wing
                    0,
                    # position
                    "",
                    int(shot.success),
                    # xG
                    float(0)))

    return arr, goals


def extract_event_data_from_game(events, gameID):
    arr = []
    for event in events:
        arr.append((int(gameID), str(event.event_type), str(event.event_description), str(event.penaltySeverity),
                    int(event.penaltyMinutes), event.playerid, event.secondaryPlayerID, event.x_coordinate,
                    event.y_coordinate, int(event.period), str(event.period_time), 0))
    return arr



def parse_shot_data(game_json):
    shots, events = parse_plays_single_game(game_json)

    shot_data = build_shot_data_single_games(game_json, shots)
    shotvals, goalvals = extract_shot_data_from_game(shot_data, game_json['gameData']['game']['pk'])



    sql = "INSERT INTO new_shot_table (PlayerID, GameID, TeamID, ShotType, Period, PeriodTime, Distance, Angle, X_coordinate, Y_coordinate, EWDiffLastEvent, NSDiffLastEvent, TimeSinceLastShiftChange, GameTimeSeconds, State, PrevEventJSON, PrevEventType, PrevEventTimeSince, PrevEventDistance, PrevEventWasFriendly, Score, IsNetEmpty, OffWing, Position, Success, xG) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    execute_table(sql, shotvals)

    return None

def execute_table(sql, vals):
    mycursor.executemany(sql, vals)
    mydb.commit()
    print(mycursor.rowcount, "record inserted.")
    return


def get_games_for_today():
    today = date.today()
    yesterday = today - timedelta(days=1)

    query = "https://statsapi.web.nhl.com/api/v1/schedule?date=" + str(yesterday)
    # query = "https://statsapi.web.nhl.com/api/v1/schedule?date=2021-10-16"
    response = requests.get(query)
    data = response.json()

    games = []
    # if there are no games for the day
    if len(data['dates']) == 0:
        return

    for game in data['dates'][0]['games']:
        print(game['gamePk'])
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        response = session.get("https://statsapi.web.nhl.com/api/v1/game/" + str(game['gamePk']) + "/feed/live")
        data = response.json()

        if data['gameData']['status']['abstractGameState'] == "Final":
            games.append(data)

    return games


def get_games_until_today():
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    start_date = date(2021, 10, 12)
    end_date = date.today()
    retrieved_games = list()
    for single_date in daterange(start_date, end_date):
        query = "https://statsapi.web.nhl.com/api/v1/schedule?date=" + str(single_date)
        # query = "https://statsapi.web.nhl.com/api/v1/schedule?date=2021-10-16"
        response = requests.get(query)
        data = response.json()

        # if there are no games for the day
        if len(data['dates']) == 0:
            continue

        for game in data['dates'][0]['games']:
            print(game['gamePk'])
            session = requests.Session()
            retry = Retry(connect=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            response = session.get("https://statsapi.web.nhl.com/api/v1/game/" + str(game['gamePk']) + "/feed/live")
            data = response.json()

            if data['gameData']['status']['abstractGameState'] == "Final":
                retrieved_games.append(data)

    return retrieved_games


def update_dates(gameid):
    sql = "select season from game where gameid = " + str(gameid)
    mycursor.execute(sql)
    seasons = mycursor.fetchall()
    for season in seasons:
        request = "https://statsapi.web.nhl.com/api/v1/schedule?season=" + season[0]
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        response = session.get(request)
        data = response.json()
        for day in data['dates']:
            dayOfGame = str(day['date'])
            for game in day['games']:
                gameID = str(game['gamePk'])
                sql = "select * from game where gameid = " + gameID
                mycursor.execute(sql)
                res = mycursor.fetchall()
                if len(res) != 0:
                    sql = "UPDATE game SET DayOfGame = \'" + dayOfGame + "\' WHERE GameID = " + gameID
                    print(sql)
                    mycursor.execute(sql)
                    mydb.commit()


def update_entire_shot_info():
    import time
    sql = "select * from new_shot_table where state = ''"
    mycursor.execute(sql)
    shots = mycursor.fetchall()
    count = 0
    for shot in shots:

        # score
        sql = 'select count(*) from gamegoal gg inner join new_shot_table st on st.gameid = gg.gameid and st.shotid = ' + str(
            shot[
                0]) + ' where gg.periodtime < st.periodtime and gg.GoalPlayerID  in (select PlayerID from (select ur.PlayerID, ur.rosterid from uroster ur inner join game g on (g.HomeRosterID = ur.rosterid or g.AwayRosterID = ur.rosterid) where g.gameid = st.gameid) a where rosterid = (select ur.rosterid from uroster ur inner join game g on (g.HomeRosterID = ur.rosterid or g.AwayRosterID = ur.rosterid) where g.gameid = st.gameid and ur.playerid = st.playerid limit 1))'
        mycursor.execute(sql)
        info = mycursor.fetchall()
        friendly_goals = int(info[0][0])

        sql = 'select count(*) from gamegoal gg inner join new_shot_table st on st.gameid = gg.gameid and st.shotid = ' + str(
            shot[
                0]) + ' where gg.periodtime < st.periodtime and gg.GoalPlayerID not in (select PlayerID from (select ur.PlayerID, ur.rosterid from uroster ur inner join game g on (g.HomeRosterID = ur.rosterid or g.AwayRosterID = ur.rosterid) where g.gameid = st.gameid) a where rosterid = (select ur.rosterid from uroster ur inner join game g on (g.HomeRosterID = ur.rosterid or g.AwayRosterID = ur.rosterid) where g.gameid = st.gameid and ur.playerid = st.playerid limit 1))'
        mycursor.execute(sql)
        info = mycursor.fetchall()
        enemy_goals = int(info[0][0])
        if info[0][0] is None:
            continue
        sql = "UPDATE new_shot_table SET Score = \'" + str(friendly_goals - enemy_goals) + "\' WHERE ShotID = " + str(
            shot[0])
        mycursor.execute(sql)

        # state
        sql = "select nst.shotid, Case WHEN nst.TeamID = gs.TeamIDUp THEN gs.State ELSE Reverse(gs.State) END as State from new_shot_table nst inner join gamestate gs on nst.periodtime between gs.timestart and gs.timeend where nst.shotid = " + str(shot[0]) + " limit 1"
        mycursor.execute(sql)
        info = mycursor.fetchall()
        if info[0][1] is None:
            continue
        state = str(info[0][1])
        sql = "UPDATE new_shot_table SET State = \'" + state + "\' WHERE ShotID = " + str(shot[0])
        mycursor.execute(sql)
        #execute_many(sql, state)

        # time diff
        sql = 'select TIME_TO_SEC(TIMEDIFF(nst.periodtime, (select s.starttime from shifts s where s.playerid = nst.playerid and s.gameid = nst.gameid and (nst.periodtime between s.starttime and s.endtime) limit 1))), nst.shotid as diff from new_shot_table nst where nst.shotid = ' + str(shot[0]) + " limit 1"
        mycursor.execute(sql)
        info = mycursor.fetchall()
        if info[0][0] is None:
            continue
        differences = int(info[0][0])
        sql = "UPDATE new_shot_table SET TimeSinceLastShiftChange = " + str(differences) + " WHERE ShotID = " + str(shot[0])
        mycursor.execute(sql)

        # shot count
        sql = 'select count(*) from new_shot_table nst inner join new_shot_table nst_2 on nst_2.TeamID = nst.TeamId and nst.gameid = nst_2.gameid and nst_2.PeriodTime between DATE_SUB(nst.periodtime, INTERVAL 10 SECOND) and nst.periodtime where nst.shotid = ' + str(shot[0])
        mycursor.execute(sql)
        info = mycursor.fetchall()
        if info[0][0] is None:
            continue
        shot_count = int(info[0][0])
        sql = "UPDATE new_shot_table SET ShotsLastTen = " + str(shot_count) + " WHERE ShotID = " + str(shot[0])
        mycursor.execute(sql)

        count = count + 1
        if count % 10000 == 0:
            print("Done with: {}".format(count))
            mydb.commit()



    """
    # state
    print("Running")
    sql = "select Case WHEN TeamID = TeamIDUp THEN State ELSE Reverse(State) END as State, shotid from (select st.shotid, (select r.teamid from game g inner join uroster ur on ur.rosterid = g.AwayRosterID or ur.rosterid = g.HomeRosterid inner join roster r on r.rosterid = ur.rosterid where ur.playerid = st.playerid and g.gameid = st.gameid limit 1) as TeamID, (select TeamIDUp from gamestate gs where st.gameid = gs.gameid and st.periodtime between gs.TimeStart and gs.TimeEnd limit 1) as TeamIDUp, (select state from gamestate gs where st.gameid = gs.gameid and st.periodtime between gs.TimeStart and gs.TimeEnd limit 1) as State from new_shot_table st) x"
    mycursor.execute(sql)
    state = mycursor.fetchall()
    sql = "UPDATE new_shot_table SET State = %s WHERE ShotID = %s"
    execute_many(sql, state)
    """
    mydb.commit()
    print("Finished updating State!")

    sql = "select * from new_shot_table where position = ''"
    mycursor.execute(sql)
    shots = mycursor.fetchall()
    count = 0
    for shot in shots:
        # position
        sql = 'select p.position, nst.shotid from new_shot_table nst inner join player p on p.playerid = nst.playerid where nst.shotid = ' + str(shot[0])
        mycursor.execute(sql)
        position = mycursor.fetchall()
        try:
            position = str(position[0][0])
        except:
            continue
        sql = "UPDATE new_shot_table SET Position = '" + str(position) + "'  WHERE ShotID = " + str(shot[0])
        mycursor.execute(sql)

        # offwing
        sql = 'select case when nst.Y_coordinate > 0 and p.handedness = \'R\' then 1 when nst.Y_coordinate < 0 and p.handedness = \'L\' then 1 else 0 end as offwing, nst.shotid from new_shot_table nst inner join player p on p.playerid = nst.playerid where nst.shotid = ' + str(shot[0])
        mycursor.execute(sql)
        offwing = mycursor.fetchall()
        if offwing[0][0] is None:
            continue
        offwing = int(offwing[0][0])
        sql = "UPDATE new_shot_table SET OffWing = " + str(offwing) + " WHERE ShotID = " + str(shot[0])
        mycursor.execute(sql)
        count = count + 1
        if count % 10000 == 0:
            print("Done with: {}".format(count))
            mydb.commit()

    mydb.commit()

    return


def execute_many(sql, vals):
    mycursor.executemany(sql, vals)
    mydb.commit()
    print(mycursor.rowcount, "record inserted.")
    return



def update_entire_shot_differential():
    sql = "select * from shottable"
    mycursor.execute(sql)
    shots = mycursor.fetchall()
    for shot in shots:
        sql = 'select count(*) from gamegoal gg inner join shottable st on st.gameid = gg.gameid and st.shotid = ' + str(
            shot[
                0]) + ' where gg.periodtime < st.periodtime and gg.GoalPlayerID  in (select PlayerID from (select ur.PlayerID, ur.rosterid from uroster ur inner join game g on (g.HomeRosterID = ur.rosterid or g.AwayRosterID = ur.rosterid) where g.gameid = st.gameid) a where rosterid = (select ur.rosterid from uroster ur inner join game g on (g.HomeRosterID = ur.rosterid or g.AwayRosterID = ur.rosterid) where g.gameid = st.gameid and ur.playerid = st.playerid limit 1))'
        mycursor.execute(sql)
        info = mycursor.fetchall()
        friendly_goals = int(info[0][0])

        sql = 'select count(*) from gamegoal gg inner join shottable st on st.gameid = gg.gameid and st.shotid = ' + str(
            shot[
                0]) + ' where gg.periodtime < st.periodtime and gg.GoalPlayerID not in (select PlayerID from (select ur.PlayerID, ur.rosterid from uroster ur inner join game g on (g.HomeRosterID = ur.rosterid or g.AwayRosterID = ur.rosterid) where g.gameid = st.gameid) a where rosterid = (select ur.rosterid from uroster ur inner join game g on (g.HomeRosterID = ur.rosterid or g.AwayRosterID = ur.rosterid) where g.gameid = st.gameid and ur.playerid = st.playerid limit 1))'
        mycursor.execute(sql)
        info = mycursor.fetchall()
        enemy_goals = int(info[0][0])
        if info[0][0] is None:
            continue
        sql = "UPDATE shottable SET Score = \'" + str(friendly_goals - enemy_goals) + "\' WHERE ShotID = " + str(
            shot[0])
        print(sql)
        mycursor.execute(sql)
        mydb.commit()
    return


def update_entire_shot_position():
    sql = "select * from new_shot_table"
    mycursor.execute(sql)
    shots = mycursor.fetchall()
    for shot in shots:
        sql = 'select p.position from new_shot_table nst inner join player p on p.playerid = nst.playerid where shotid = '+ str(shot[0])
        mycursor.execute(sql)
        info = mycursor.fetchall()
        player_position = info[0][0]
        if info[0][0] is None:
            continue
        sql = "UPDATE new_shot_table SET Position = \'" + str(player_position) + "\' WHERE ShotID = " + str(shot[0])
        print(sql)
        mycursor.execute(sql)
        mydb.commit()
    return

def update_entire_offwing():
    sql = "select * from new_shot_table"
    mycursor.execute(sql)
    shots = mycursor.fetchall()
    for shot in shots:
        sql = 'select case when nst.Y_coordinate > 0 and p.handedness = \'R\' then 1 when nst.Y_coordinate < 0 and p.handedness = \'L\' then 1 else 0 end as offwing from new_shot_table nst inner join player p on p.playerid = nst.playerid where shotid = ' + str(shot[0])
        mycursor.execute(sql)
        info = mycursor.fetchall()
        offwing = info[0][0]
        if info[0][0] is None:
            continue
        sql = "UPDATE new_shot_table SET OffWing = \'" + str(offwing) + "\' WHERE ShotID = " + str(shot[0])
        print(sql)
        mycursor.execute(sql)
        mydb.commit()
    return

def update_entire_shot_count():
    sql = "select * from new_shot_table"
    mycursor.execute(sql)
    shots = mycursor.fetchall()
    count = 0
    for shot in shots:
        # shot count
        sql = 'select count(*) from new_shot_table nst inner join new_shot_table nst_2 on nst.shotid != nst_2.shotid and nst_2.TeamID = nst.TeamId and nst.gameid = nst_2.gameid and nst_2.PeriodTime between DATE_SUB(nst.periodtime, INTERVAL 10 SECOND) and nst.periodtime where nst.shotid = ' + str(
            shot[0])
        mycursor.execute(sql)
        info = mycursor.fetchall()
        if info[0][0] is None:
            continue
        shot_count = int(info[0][0])
        sql = "UPDATE new_shot_table SET ShotsLastTen = " + str(shot_count) + " WHERE ShotID = " + str(shot[0])
        print(sql)
        mycursor.execute(sql)
        count = count + 1
        if count % 10000 == 0:
            print("Done with: {}".format(count))
            mydb.commit()
    mydb.commit()
    return



if __name__ == "__main__":


    """
    #2007 does not seem to work.... need to figure out why
    year = 2013
    while year <= 2020:
        games = get_regular_season(str(year))
        for game in games:
            parse_shot_data(game)
        year = year + 1

    #update_entire_shot_position()
    #update_entire_offwing()
    """
    #update_entire_shot_info()
    update_entire_shot_count()


    """
    games = get_games_for_today()
    for game in games:
        population_just_event_tables(game)
    """

    """
    games = get_games_for_today()
    # games = get_games_until_today()

    if len(games) > 0:
        for game in games:
            sql = "select * from game where gameid = " + str(game['gameData']['game']['pk'])
            mycursor.execute(sql)
            exist = mycursor.fetchall()
            if len(exist) == 0:
                parse_shot_data(game)
                update_new_shot_differential(game['gameData']['game']['pk'])
        update_dates(games[0]['gameData']['game']['pk'])
    """






