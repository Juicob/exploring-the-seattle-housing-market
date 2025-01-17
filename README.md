# Project

This project was created in order to provide my wife an I a better idea of where to buy a home in Seattle. There was lots of housing data provided to us for the project; however, there were a few big issues that needed to be first dealt with before I was able to answer the questions I had. I focused a lot on location during this project because that and price are one of our top priorities we look at when purchasing a home. Having lived in Seattle before, I was familiar with the neighborhoods and had an idea of where we'd like to be, but wanted to use the data to bring out insights and see if it aligned with my assumptions. To do this, I discovered, and learned how to use the QGIS platform in order to gather the data I needed which was an adventure in itself!

Due to a git snafu which was definitely a learning experience, I lost my actual power point file. I do still have the [video presentation](https://drive.google.com/file/d/1pSBoMdj5hw3OgT00YLVCczf-W-REhi2p/view?usp=sharing) of myself speaking over my slides though so feel free to take a look! Please note that this is before I added my final thank you slide!

# Built With

- Python
- QGIS
- Plotly
- Map Box
- Seaborn
- Pandas
- Geopy
- Numpy
- Sci-kit Learn
- Open Brewery Database

# Question 1

## Which houses are within a 10 minute drive from the hospital?

### Thoughts
Nobody likes a commute! Since I primarily work from home, my commute is measured more in a distance of feet rather than miles. My wife however prefers to live relatively close to work so she can respond quickly while on call and would prefer to be close enough to walk or ride a bike to work as a way to destress before or after a hard shift. 

The data given only provided coordinates of the houses so I found the coordinates of the hospital (University of Washington Medical Center). After that I also found a dataset from the Seattle Network Database which had all of the major roads within the Seattle proper area that I was able to import into QGIS. Overlaying the houses given from the dataset I'm then able to perform a network analysis with the two layers that calculates the drive time it takes to get from each house to the the coordinates given that represents the hospital. This calculation is done at a speed of about 31 miles per hour which I believe is underestimating the speed limit for paths outside of busy areas but thought it would make up for potential rush-hour delays.

<table align="center">
<tr>
<td style="width:300px; height:300px; text-align:center; vertical-align:middle">
<img src='https://i.imgur.com/dvE3Hvs.png' style="display:block;" width="300" height="300"/>
</td>
<td style="width:300px; height:300px; text-align:center; vertical-align:middle">
<img src='https://i.imgur.com/F719k70.png' style="display:block;" width="300" height="300"/>
</td>
<td style="width:300px; height:300px; text-align:center; vertical-align:middle">
<img src='https://i.imgur.com/FFYLpHa.png' style="display:block;" width="300" height="300"/>
</td>
<td style="width:300px; height:300px; text-align:center; vertical-align:middle">
<img src='https://i.imgur.com/kpRqyui.png' style="display:block;" width="300" height="300"/>
</td>
</tr>
</table>

### Findings

Great! Based on the homes within the map above we've got a separate dataframe full of houses that are within 10 minutes away from the hospital. Even though we're open to that distance, the closer to work for her the better. I'm not too familiar with the Ravenna or Laurelhurst neighborhoods but based on the map it seems like the homes in Larelhurst are much more expensive than Ravenna and also Wallingford which might cause us to lean to those areas more.  

# Question 2

## Which houses are within a 2 mile radius from a few of my favorite breweries?

### Thoughts

Now that my wife is covered, I want to see which of these are close by to some of my favorite breweries in Seattle. We both enjoy eating out and trying new restuarants and breweries and many of the breweries are located within cool parts of town that we tend to find ourselves in anyways so if we can find a home that is both close to her work and in a part of town that we enjoy that would be ideal!

The breweries I picked were completely based off of personal preference. We've shared many memories at these locations or with the products that they've offered and when we think about when we used to live in Seattle, taprooms and beer gardens are where we see ourselves. Except for Optimism. Their beer kinda sucks but they have a beautiful space. Honestly I only chose that because I felt like I had to represent the Cap hill neighborhood and wanted to stick with "breweries" and not drift off to "bars" or "taprooms; otherwise, Pine Box would've been the move.

First I had to find the locations of a few of my favorite breweries. I could've done this manually just through google maps, but I thought it'd be good practice to put OBDB (Open Brewery Database) to use. I connected to the api and gathered data including the coordinates of those breweries and created a dataframe out of it. I then plotted it along with the houses, and used Geopy to calculate the distance in miles from each house to each brewery. After that I created a column that represented the brewery each house was closest to and that distance. I was then able to plot these on a map to visualize the housing options we had that were within 2 miles of these 6 breweries around Seattle. 

<table align=center>
<tr>
    <td>
<img src='https://i.imgur.com/02kQHuW.png' height = 400 width= 400/>
    </td>
    <td>
<img src='https://i.imgur.com/pIPf2wF.png' height = 400 width= 400/>
    </td>
    </tr>
</table>
</p>

### Findings

Good thing that the hospital is located where it is! It looks like there's plenty of overlap in houses that fall within both 10 minutes of driving distance to the hospital and a 2 mile radius of some of my favorite breweries! Based on these maps and only these two variables, it seems like we'd try to look for a house in the Fremont or Wallingford neighborhoods, which actually falls in line to where we'd like to be! 

The hospital is located in the University District which is filled with student housing and not exactly a place we want to live at this point in our lives. Fremont is kind of a trendy/artsy neighborhood with fantastic food and drink options for us to enjoy!

# Question 3

## Does a house position within 1/4 mile of the water affect the price of homes?

### Thoughts
With so much water around Seattle, I would imagine there are several properties that would qualify as "waterfront properties". I wanted to know if being close to the water had a significant impact on the price. The given dataset included a "waterfront" column marked with 1's or 0's to denote if a home is or is not a waterfront property respectively. After exploring the data, I realized that there seems to be an inconsistency within the data. The data doesn't seem to properly reflect homes that are on the waterfront, and seems to be missing several homes that should be labeled as "waterfront" but are not. In order to fix this problem, I was able to source a dataset from the USGS (United States Geological Survey) that displayed the bodies of water around Seattle. Importing this data as a layer within QGIS, I was then able to find the distance of each house to the nearest body of water. One of the issues I had initially was the prevalence of small ponds or lakes around King County. I decided to narrow down qualifying bodies of water to only large lakes and bays (including Greenlake) as I don't believe having a house on with a view of a small pond is an incredibly valued feature. Below, the first image shows the original file I started with showing all the small lakes that I needed to remove. The map is slightly more zoomed out and the bodies of water are colored in blue to be more reconizable. The second image shows a map of each house and a line that represents the distance from that house to the nearest body of water. This is before removing the small lakes and ponds from the dataset. The third image shows the lines from each house to each major body of water, and I removed the dots for each house to get a cleaner view.

<br/>

<table>
    <tr>
        <td style="width:300px; height:300px; text-align:center; vertical-align:middle">
<img src='https://i.imgur.com/94Ghx0p.png' style="display:block;" width="300" height="300"/>
        </td>
        <td style="width:300px; height:300px; text-align:center; vertical-align:middle">
            <img src='https://i.imgur.com/j3IdkzK.png' style="display:block;" width="300" height="300"/>
        </td>
        <td style="width:300px; height:300px; text-align:center; vertical-align:middle">
<img src='https://i.imgur.com/m3jlQAr.png' style="display:block;" width="300" height="300"/>
        </td>
    </tr>
</table>

<br/>
<br/>

After exporting that distance data from QGIS, I was able to work with it in Python using the same Plotly/Mapbox libraries I used for the first two quesetions. The first image shows the initial representation of waterfront homes with the given dataset while the second image displays another map showing each house with colored by it's distance away from the closest body of water. This map wasn't the most informative. There's so much water around the Seattle area that the vast majority of homes are within 5 miles and so most of the homes are colored dark blue. There were houses over 30 miles away in the South East that are out of frame and you can see the shade of dark blue start turning purple but it still didn't offer as much information as I'd prefer. I decided to define a waterfront property as being within 1/4 mile away from the shore and rebuild our map based on this metric which told a much better story.

<br/>
<br/>

<table>
    <tr>
        <td style="width:300px; height:300px; text-align:center; vertical-align:middle">
            <img src='https://i.imgur.com/0s2GCng.png' style="display:block;" width="300" height="300"/>
        </td>
        <td style="width:300px; height:300px; text-align:center; vertical-align:middle">
<img src='https://i.imgur.com/7LBUdQH.png' style="display:block;" width="300" height="300"/>
        </td>
        <td style="width:300px; height:300px; text-align:center; vertical-align:middle">
<img src='https://i.imgur.com/KDbcnkb.png' style="display:block;" width="300" height="300"/>
        </td>
    </tr>
    </table>
</div>

### Findings
Much better. Now our data more accurately represents what is and isn't a waterfront property and we can also zoom in and visually inspect some information about each home. Just browing the map it does look like that houses along the waterfront tend to be more expensive than those more inland. However there are exceptions to that most notably around the Lower Queen Anne, Capitol Hill, and Ashwood neigborhoods. Off of memory I know that Queen Anne is very close to South Lake Union which is where the Amazon headquarters is, Capitol Hill is a very trendy and busy part of town, and Ashwood is relatively close to where Microsofts headquarters is it makes a bit of sense to me that these areas might have more expensive homes. 

After this, I conducted a hypothesis test that proved that a house located within 1/4 mile from the shore does have an affect on price.

# Linear Regression

Using a linear regression model I was able to explain over 90% of the variance of price using the square feet of living space, the ratio of bedrooms to bathrooms, and the distance from the closest of 6 breweries. Below are the results of my OLS model and the normality and homoskedasticity of the resulting residuals.

<br/>
<br/>

<table>
    <tr>
        <td style="width:300px; height:300px; text-align:center; vertical-align:middle">
<img src='https://i.imgur.com/mBKpFBg.png' style="display:block;" width="300" height="300">
        </td>
        <td style="width:300px; height:300px; text-align:center; vertical-align:middle">
<img src='https://i.imgur.com/zF4xI39.png' style="display:block;" width="300" height="300">
        </td>
        <td style="width:300px; height:300px; text-align:center; vertical-align:middle">
<img src='https://i.imgur.com/ihJZdZW.png' style="display:block;" width="300" height="300">
        </td>
    </tr>
</table>
        

# Future Work

If I had more time - 

- I'd like to apply the same methodology I used for travel time to the hospital to all 6 of the breweries as well but instead calculate for walking time instead of driving time. Once I have that calculation for all the houses to their closest breweries. I'd like to map it out by binning the walking time in 5 minute increments around the breweries so that I'm able to toggle on and off the walking time of all the houses within my maps. 

- I'd also like to see how the prevalance of farmers markets affect housing prices within the area. My wife and I would frequent the Ballard and University Districts Farmers Markets and still reminese over the smell of fresh donuts and delightfully salty cheeses we would pick up over the weekend, not to mention how hard it was to find parking. There are a number of large farmers markets in Seattle and I think it would be interesting to see how the proximity of houses to these hubs of sustainability and hipster fare affect housing prices within the area.
