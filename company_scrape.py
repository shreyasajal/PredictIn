from selenium import webdriver
from time import  sleep 
from bs4 import BeautifulSoup
from random import randint
import pandas as pd 
import numpy as np 

driver = webdriver.Chrome()

driver.set_page_load_timeout(90)

driver.get("https://www.linkedin.com")
sleep(2)

profile = driver.find_element_by_id('session_key')
profile.send_keys("")
sleep(0.5)

pswd = driver.find_element_by_id('session_password')
pswd.send_keys("")
sleep(0.5)

log_in = driver.find_element_by_class_name('sign-in-form__submit-button')
log_in.click()
sleep(2)

profiles = pd.read_csv("CompanyProfiles.csv")

columns = ["name","headline","location","followers","connections","about","time_spent","content","content_links","media_type","media_urls","num_hashtags","hashtag_followers","hashtags","reactions","comments","views","votes"]

scraped_data = pd.DataFrame(columns=columns)

for profile in profiles.iloc[66:69,2]:

    driver.get(profile+"about/")
    sleep(2)

    soup = BeautifulSoup(driver.page_source,'lxml')

    acc_name = soup.find(class_="org-top-card-summary__title")
    acc_name = acc_name.get_text().strip() if acc_name else "N/A"
    # print(acc_name)

    info = soup.find_all(class_="org-top-card-summary-info-list__info-item")
    headline = info[0].get_text().strip() if len(info)==3 else "N/A"
    location = info[1].get_text().strip() if len(info)==3 else "N/A"
    followers = info[2].get_text().strip() if len(info)==3 else "N/A"

    # print(headline)
    # print(location)
    # print(followers)

    about = soup.find(class_="white-space-pre-wrap")
    about = about.get_text().strip() if about else "N/A"
    # print(about)

    emp_linkedin = soup.find(class_="org-page-details__employees-on-linkedin-count")
    emp_linkedin = emp_linkedin.text.strip().split()[0] if emp_linkedin else "N/A"
    # print(emp_linkedin)

    driver.get(profile+"posts/?feedView=all")
    sleep(2)

    SCROLL_PAUSE_TIME = 20

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    soup = BeautifulSoup(driver.page_source,'lxml')

    posts = soup.find_all("div",class_="occludable-update")
    total_posts = len(posts)

    for post in posts:

        feed_type = post.find("div")
        feed_type = feed_type["data-urn"].split(":")[2] if feed_type else "N/A"
        # print(feed_type)

        if feed_type == "activity":

            time_spent = post.find(class_="feed-shared-actor__sub-description").find("span",class_="visually-hidden")
            time_spent = time_spent.get_text().strip() if time_spent else "N/A"
            # print(time_spent)

            content = post.find(class_="feed-shared-update-v2__description-wrapper")

            content_links = []
            if content:
                for link in content.find_all("a",href=True):
                    content_links.append([link["href"],link.text])
                    # print(link["href"])
                    # print(link.text)

            content = content.get_text(" ").strip() if content else "N/A"
            # print(content)

            interactions = post.find(class_="social-details-social-counts")
            interactions = interactions.get_text().strip().split() if interactions else "N/A"
            # print(len(interactions))

            reactions=0
            comments=0
            views="N/A"
            votes = "N/A"

            if interactions != "N/A":
                for i in range(len(interactions)):
                    if(i==0): 
                        reactions = int(interactions[i].replace(",",""))
                    if(interactions[i]=="comment" or interactions[i]=="comments"):
                        comments = int(interactions[i-1].replace(",",""))
                        if(i==1):
                            reactions=0
                    if(interactions[i]=="view" or interactions[i]=="views"):
                        views = interactions[i-1]
                        if(i==1):
                            reactions=0

            # print(reactions)
            # print(comments)
            # print(views)

            media = post.find(class_="feed-shared-update-v2__content")
            media_type = "N/A"
            media_url = "N/A"

            if media:
                media_type = media["class"][1].split("-")[-1]
                if media_type == "hidden":
                    media_type = media["class"][2].split("-")[-1]
                if media_type == "image":
                    media_url = ""
                    for image in media.find_all("img"):
                        media_url = media_url + image["src"] + " " if image else media_url
                    media_url = media_url.strip()
                if media_type == "document":
                    media_url = ""
                    for image in media.find_all("img"):
                        media_url = media_url + image["data-src"] + " " if image else media_url
                    media_url = media_url.strip()
                if media_type == "video":
                    media_url = media.find("video")
                    media_url = media_url["src"] if media_url else "N/A"
                if media_type == "article":
                    media_url = media.find("a")
                    media_url = media_url["href"] if media_url else "N/A"
                if media_type == "poll":
                    votes = media.find(class_="feed-shared-poll-summary__subtext-container")
                    votes = votes.find("p") if votes else votes
                    votes = votes.get_text().strip().split()[0] if votes else "N/A"
                # print(media_url)
                # print(media_type)

            media_urls = media_url.strip().split() if media_url!="N/A" else []

            num_hashtags = 0
            hashtag_followers = 0
            hashtags = []

            for link in content_links:
                if link[1][0]=="#":
                    num_hashtags+=1
                    # driver.get(link[0])
                    # sleep(1)
                    # soup = BeautifulSoup(driver.page_source,'lxml')
                    # hf = int(soup.find(class_="feed-hashtag-feed__artdeco-card").find("p").get_text().strip().split()[0].replace(",",""))
                    # hashtag_followers+=hf
                    hashtags.append([link[0],link[1]])

            # # print(num_hashtags)
            # # print(hashtag_followers)

            scraped_data = scraped_data.append({"name":acc_name,"headline":headline,"location":location,"followers":followers,
                                                "connections":emp_linkedin,"about":about,"time_spent":time_spent,
                                                "content":content,"content_links":content_links,"media_type":media_type,
                                                "media_urls":media_urls,"num_hashtags":num_hashtags,
                                                "hashtag_followers":hashtag_followers,"hashtags":hashtags,"reactions":reactions,
                                                "comments":comments,"views":views,"votes":votes},ignore_index=True)

        else:
            continue
    
    os.remove("company_data.csv") if os.path.exists("company_data.csv") else None
    scraped_data.to_csv("company_data.csv")
    print(acc_name+" had "+str(total_posts)+" posts")