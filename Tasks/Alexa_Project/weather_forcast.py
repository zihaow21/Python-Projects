import pywapi

class WeatherForecast(object):
    def __init__(self, location):
        self.location = location

    def weatherInfo(self):
        loc_id = pywapi.get_loc_id_from_weather_com(self.location)[0][0]
        data = pywapi.get_weather_from_weather_com(loc_id, units='imperial')
        conditions = data['current_conditions']
        temperature = conditions['temperature']
        feels_like = conditions['feels_like']
        uv_level = conditions['uv']["text"]
        wind_speed = conditions["wind"]["speed"]
        if eval(wind_speed) == 1:
            wind_speed = "1 mile per hour"
        else:
            wind_speed = "{} miles per hour".format(eval(wind_speed))
        text = conditions["text"]
        visibility = conditions["visibility"]
        if eval(visibility) == 1.0:
            visibility = "1 mile"
        else:
            visibility = "{} miles".format(eval(visibility))

        forecasts = data["forecasts"]
        day = []
        temp_high = []
        temp_low = []
        brief_text_day = []
        chance_precipitation_day = []
        chance_precipitation_night = []
        brief_text_night = []
        date = []
        wind_text_day = []
        wind_text_night = []
        sun_rise = []
        sun_set = []

        for i in range(len(forecasts)):
            f = forecasts[i]
            day.append(f["day_of_week"])
            date.append(f["date"])
            temp_high.append(f["high"])
            temp_low.append(f["low"])
            brief_text_day.append(f["day"]["brief_text"])
            chance_precipitation_day.append(f["day"]["chance_precip"])
            wind_text_day.append(f["day"]["wind"]["text"])
            chance_precipitation_night.append(f["night"]["chance_precip"])
            brief_text_night.append(f["night"]["brief_text"])
            wind_text_night.append(f["night"]["wind"]["text"])
            sun_rise.append(f["sunrise"])
            sun_set.append([f["sunset"]])

        current_info= "Yes, of course. . . The overall weather is {}. The current temperature is {}. Feels like {}. " \
                      "Wind speed is {}. Visibility is {}. UV level is {}.".format(text, temperature, feels_like,
                                                                                   wind_speed, visibility, uv_level)
        forcasts_info = "Sure. The forecasts for the following days are"
        for i in range (len(forecasts)):
            forcasts_info += " {}, {}, temperature high {}, temperature low {}, {}, " \
                            "{} wind, chance of precipitation {} percent. sun rise {}, sun set {}, {}" \
                            "during the night, chance of precipitation {}, {} wind".format(day[i], date[i],
                            temp_high[i], temp_low[i], brief_text_day[i], wind_text_day[i], chance_precipitation_day[i],
                            sun_rise[i], sun_set[i], brief_text_night[i], chance_precipitation_night[i], wind_text_night[i])
        return current_info, forcasts_info



# wf = WeatherForecast("atlanta Georgia")
# string = wf.weatherInfo()