from selenium import webdriver
from time import  sleep 
from bs4 import BeautifulSoup
from selenium.webdriver.common.action_chains import ActionChains





driver = webdriver.Chrome()

driver.set_page_load_timeout(90)

driver.get("https://www.linkedin.com")

profile = driver.find_element_by_id('session_key')
profile.send_keys("")
sleep(0.5)

pswd = driver.find_element_by_id('session_password')
pswd.send_keys("")
sleep(0.5)

log_in = driver.find_element_by_class_name('sign-in-form__submit-button')
log_in.click()
sleep(2)
profiles = pd.read_csv("id.csv")

columns = ["name","headline","location","followers","connections","about","time_spent","content","content_links","media_type","media_url","num_hashtags","hashtag_followers","hashtags","reactions","comments","views","votes"]
scraped_data = pd.DataFrame(columns=columns)
for profile in profiles['           Linkedin_Id'][:33]:
    driver.get(profile)
    sleep(4)
    sp=BeautifulSoup(driver.page_source,'lxml')
    e=driver.find_element_by_id('line-clamp-show-more-button')

    if e:
    
        e.click()
        sleep(0.5)
        soup0=BeautifulSoup(driver.page_source,'lxml')
    else:
        driver.execute_script("window.scrollTo(0, 2100)")
        soup0=BeautifulSoup(driver.page_source,'lxml')
    
    page = driver.get(profile+ 'detail/recent-activity/shares/')
    sleep(5)
    SCROLL_PAUSE_TIME = 5

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
    soup= BeautifulSoup(driver.page_source,'lxml')
    posts=soup.find_all('div',class_="occludable-update")
    acc_name=soup.find('h3',class_='single-line-truncate t-16 t-black t-bold mt2')
    acc_name=soup.find('h3',class_='single-line-truncate t-16 t-black t-bold mt2').text.strip() if name else "N/A"

    info=sp.find(class_="pv-top-card--list pv-top-card--list-bullet mt1")
    info=sp.find(class_="pv-top-card--list pv-top-card--list-bullet mt1").text.strip().split() if info else "N/A"
    location="N/A"
    connections="N/A"
    
    if info[1]=='connections':
        location='N/A'
        connections=info[0]
    else:
        b=info.index('connections')
        location=info[:b-1]
        connections=info[b-1]
        
    about=soup0.find(class_="lt-line-clamp__raw-line")
    about=soup0.find(class_="lt-line-clamp__raw-line").text.strip() if about else "N/A"
    followers=soup.find('div',class_='display-flex justify-space-between t-12 t-bold t-black--light mt3')
    followers=soup.find('div',class_='display-flex justify-space-between t-12 t-bold t-black--light mt3').text.strip().split()[1].replace(',','') if followers else "N/A"
    headline=soup.find('h4',class_='t-14 t-black--light t-normal mb1')
    headline=soup.find('h4',class_='t-14 t-black--light t-normal mb1').text.strip() if headline else 'N/A'
    
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
            else:
                content_links = []
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
            media_urls = media_url.strip().split() if media_url!="N/A" else []
            if media:
                media_type = media["class"][1].split("-")[-1]
                if media_type == "hidden":
                    media_type = media["class"][2].split("-")[-1]
                if media_type == "article":
                    media_url = media.find("a")
                    media_url = media_url["href"] if media_url else "N/A"
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
            #         driver.get(link[0])
            #         sleep(1)
            #         soup = BeautifulSoup(driver.page_source,'lxml')
            #         hf = int(soup.find(class_="feed-hashtag-feed__artdeco-card").find("p").get_text().strip().split()[0].replace(",",""))
            #         hashtag_followers+=hf
                    hashtags.append([link[1],link[0]])

            # # print(num_hashtags)
            # # print(hashtag_followers)

            scraped_data = scraped_data.append({"name":acc_name,"headline":headline,"location":location,"followers":followers,
                                                "connections":connections,"about":about,"time_spent":time_spent,
                                                "content":content,"content_links":content_links,"media_type":media_type,
                                                "media_url":media_urls,"reactions":reactions,"comments":comments,
                                                "views":views,"votes":votes,'hashtag_followers':hashtag_followers,"num_hashtags":num_hashtags,'hashtags':hashtags},ignore_index=True)

        else:
            continue
    
    
    os.remove("influencers.csv") if os.path.exists("influencers.csv") else None
    scraped_data.to_csv("influencers.csv")
    print(acc_name)