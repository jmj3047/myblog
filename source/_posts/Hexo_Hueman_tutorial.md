---
title: Hexo Hueman Tutorial
date: 2022-04-21
categories:
  - Setting
tags: 
  - Hueman
  - Hexo
---


##### 1.[Starting Hexo Blog](https://futurecreator.github.io/2016/06/14/get-started-with-hexo/)

```bash
username@LAPTOP-D1EUIRLS MINGW64 ~/Desktop
$ hexo init your_blog_folder

username@LAPTOP-D1EUIRLS MINGW64 ~/Desktop
$ cd your_blog_folder/

username@LAPTOP-D1EUIRLS MINGW64 ~/Desktop/your_blog_folder
$ echo "# your_blog_folder" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M master
git remote add origin https://github.com/your_id/your_blog_folder.git
git push -u origin master

username@LAPTOP-D1EUIRLS MINGW64 ~/Desktop/your_blog_folder (master)
$ git add .

username@LAPTOP-D1EUIRLS MINGW64 ~/Desktop/your_blog_folder (master)
$ git commit -m "updated"

username@LAPTOP-D1EUIRLS MINGW64 ~/Desktop/your_blog_folder (master)
$ git push

username@LAPTOP-D1EUIRLS MINGW64 ~/Desktop/your_blog_folder (master)
$ code .
```



##### 2.[Applying Hueman Theme](https://futurecreator.github.io/2016/06/14/hexo-apply-hueman-theme/)



##### 3.[Basic Hexo Tutorial](https://futurecreator.github.io/2016/06/21/hexo-basic-usage/)



##### 4.[Hexo Tag Plugins](https://futurecreator.github.io/2016/06/19/hexo-tag-plugins/)


##### 5.Add Math Formula(without changing from Notion)

1) Creat File name `mathjax.ejs` on `themes/hueman/layout` folder

```python
MathJax.Hub.Config({
    jax: ["input/TeX", "output/HTML-CSS"],   #     mathjax       
    tex2jax: {
        inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']],
        processEscapes: true,
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    },
    messageStyle: "none",
    "HTML-CSS": { preferredFont: "TeX", availableFonts: ["STIX","TeX"] }
});

```
2) Check ``#Plugins`` in `themes/hueman/_config.yml` file and change `mathjax: false` to `true`
3) Add `mathjax:true` at the header when you post

- Reference: [Math Formula](https://intrepidgeeks.com/tutorial/add-hexo-mathjax-support)




##### 6.[Font Change](https://futurecreator.github.io/2018/06/12/hexo-change-font-on-hueman-theme/)


##### 7.[Deleting Posts](https://futurecreator.github.io/2017/01/15/how-to-delete-post-in-hexo/)

##### 8.[Error in Hueman Theme](https://pictureyou-neo.github.io/categories/2-Web/Hexo/)


`To Be Continued..`