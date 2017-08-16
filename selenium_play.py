from selenium import webdriver
import chromedriver_binary

driver = webdriver.Chrome()
driver.get('https://play.google.com/store/apps')
#driver.find_elements_by_link_text('カテゴリ')
#driver.find_elements_by_xpath("//div[@class = 'action-bar-dropdown']")
category_dropdowns = driver.find_elements_by_xpath("//button[@aria-controls = 'action-dropdown-children-カテゴリ']")
category_dropdowns[0].click()
subcategories = category_dropdowns[0].find_elements_by_xpath("//descendant::a[@class = 'child-submenu-link']")
subcategories[0].get_attribute("href")
driver.find_elements_by_
driver.close()