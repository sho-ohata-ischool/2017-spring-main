# Assignment 0:  Hello W266!

This assignment is a quick walk-through to help you get set up logistically for the course.

## (5 points total) Due: January 15, 11:59p Pacific

If you haven't yet, please:

- Sign up for [Piazza](http://piazza.com/berkeley/spring2017/datasciw266)
- Fill out the [Office Hours scheduling poll](http://doodle.com/poll/679ybxc3w96umfpu)
- (Monday class only) Please let the instructors know (use a private post on Piazza) *if you cannot attend class on January 16 & February 20*.  These are technically academic & administrative holidays.  We'll want to make other arrangements for you to get help with those week's material!

Now we'll get you all set up with the software packages and the course GitHub.

1. **Set up your computing environment:** We recommend using a Google Cloud Compute instance. We have $50 of credits available per student, and it should only take a few minutes to set up by following [our Cloud guide](cloud/).  
If you prefer to work on your own laptop/desktop/server (including AWS), we strongly recommend Ubuntu 16.04 with Anaconda 4.2.0 as your Python distribution. Be sure to install **TensorFlow version 0.12.1**, as the TensorFlow APIs tend to break between versions.  
*(Note that due to the variety of systems out there, we can only provide official support for Google Cloud instances.)*

2. **Clone the course repo** (if you didn't already in the Cloud guide) with:  
`git clone https://github.com/datasci-w266/2017-spring-main.git ~/w266`  
You may also want to do this on your laptop to have a local copy.  

3. **Edit the `username`** file in the `assignment/a0` folder and replace the contents with your *username@ischool.berkeley.edu* email address. *(This helps us link people to their GitHub accounts.)*  
Commit the change with `git commit`.

4. **Open and run `a0.ipynb`**. This notebook will check that your Python packages are up-to-date, test TensorFlow, and give a taste of some of the NLP datasets we'll be working with. You don't need to write any code here - just run the cells, save, and commit the updated notebook to git.

4. **Create your personal submission repo** at [this link](https://classroom.github.com/assignment-invitations/8bb12cea693c37496b7ba934bd9183e6). We'll use this for submitting assignments; it's private to you and the instructors.

5. **Run the submit script**: `./assignment/submit.sh your-github-username`, which will push to your personal repo. It will try to verify the submission, but you can also visit the repo on GitHub and confirm that your changes show up.

## Next...

Continue on to [Assignment 1](../a1/).  (Unlike Assignment 0, Assignment 1 isn't just a setup exercise.  You'll want the full two weeks to work on it.  Don't wait until next week to get started!)
