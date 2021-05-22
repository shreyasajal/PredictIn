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
hashtags = pd.read_csv("Hashtag.csv")
columns=["hashtag","followers"]
scraped_data = pd.DataFrame(columns=columns)
j=0
for i in range():
    hashtag=hashtags['Link'][i][2:-1]
    driver.get(hashtag)
    sleep(4)
    hf=0
    j=j+1
    soup = BeautifulSoup(driver.page_source,'lxml')
    hasht=soup.find(class_="feed-hashtag-feed__artdeco-card").find("p").get_text().strip().split()[0].replace(",","")
    if hasht=='Be':
        hf=0
    else:
        hf=int(hasht)
    
    scraped_data=scraped_data.append({'hashtag':hashtags['Hashtags'][i],'followers':hf},ignore_index=True)
    
    if j==200:            
        os.remove("hash_data.csv") if os.path.exists("hash_data.csv") else None
        scraped_data.to_csv("hash_data.csv")
        j=0
os.remove("hash_data.csv") if os.path.exists("hash_data.csv") else None
scraped_data.to_csv("hash_data.csv")