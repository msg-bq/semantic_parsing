import random
from mimesis import Generic, BaseProvider, BaseDataProvider
from mimesis.locales import Locale

from faker import Faker
from faker.providers import BaseProvider as FakerProvider

_CUSTOM_CONCEPTS_MAPPING = {'location': 'address'}  # 同义词的映射，比如虽然没有location函数，但不妨用address函数代替


class Weather_Temperatur_Provider(FakerProvider):
    @staticmethod
    def weather_temperature_unit() -> str:
        temperature_unit = ["c", "Celcius", "celcius", "Fahrenheit", "fahrenheit", "F", "f", "C"]
        return random.choice(temperature_unit)


class Amount_Provider(FakerProvider):
    @staticmethod
    def amount() -> str:
        amount_unit_lst = ["all", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                           "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]
        return random.choice(amount_unit_lst)


class Price_Provider(FakerProvider):
    @staticmethod
    def price_adj() -> str:
        # ,"less than $10"
        price_adj_lst = ["free", "Free", "small", "low cost", "cheap", "cheapest"]
        return random.choice(price_adj_lst)


class Ordinal_Provider(FakerProvider):
    @staticmethod
    def ordinal() -> str:
        ordinal_numbers = ["first", "second", "third", "fourth", "fifth"]
        return random.choice(ordinal_numbers)

    @staticmethod
    def future_indication_word() -> str:
        ordinal_numbers = ["next", "Next", "coming up", "upcoming", "soonest", "new"]
        return random.choice(ordinal_numbers)


class Location_Provider(FakerProvider):
    @staticmethod
    def location_adj() -> str:
        ordinal_numbers = ["outside", "inside", "outdoor", "indoor", "local", "locally"]
        return random.choice(ordinal_numbers)


class Event_Attribute_Provider(FakerProvider):
    @staticmethod
    def event_attribute() -> str:
        family_lst = ["families"]
        animals = ["pet", "dog", "cat"]
        final_animals = []
        for animal in animals:
            final_animals.append(animal)
            final_animals.append("my " + animal)
            final_animals.append(animal + "friendly")
        persons = ["family", "kid", "kids", "teen", "teens", "proteen", "proteens", "baby", "children", "adult",
                   "adults", "tooddler", "tooddlers", "handicapped", "vegetarian", "small children"]
        final_persons = []
        for person in persons:
            final_persons.append(person)
            final_persons.append("for " + person)
            final_persons.append("the " + person)
            final_persons.append(person + "only")
            final_persons.append(person + "friendly")
            final_persons.append(person + "free")
            final_persons.append(person + " - friendly")
        special_person = ["single women", "single man", "tourists", "volunteer", "youth", "runners", "baptist",
                          "couples", "young people", "volunteer", "young knitters"]
        event_attribute_adj = ["senior", "underage", "christian", "organized", "open", "popular", "weekly",
                               "educational", "special", "newest", "new", "elderly"]
        event_attribute_noun = ["alcohol", "food", "gluten", "all ages"]

        all_lst = [family_lst, final_animals, final_persons, special_person, event_attribute_adj, event_attribute_noun]
        return random.choice(random.choice(all_lst))


class Reminder_Provider(FakerProvider):

    @staticmethod
    def contact_related() -> str:
        return "my"

    @staticmethod
    def reminder_event_name() -> str:
        reminder_event_name_lst = ["Rolling Stone", "bruce", "Of Mice of Men", "JoJo", "New Edition", "Miranda Lambert",
                                   "miguel", "tyler perry", "Chris Tomlin", "Michigan Zoological Society",
                                   "The Pretenders", "Selena Gomez", "waylon jennings", "Led Zeplin",
                                   "Philharmonic Orchestra", "Hamilton", "fireline", "rihanna", "the Temptations",
                                   "beyonce", "Sia", "SIA", "Linkin Park", "Metalica", "Garth Brooks",
                                   "the Pittsburgh symphony", "the Rockets", "Keith Urban", "Harlem Boys Choir",
                                   "Macys", "Madonna", "the Shook Twins", "Beyonce", "coolio", "Lady Antebellum", "swv",
                                   "Macon Fishly", "Oingo Boingo", "Kylie Jenner", "Andre Rieu", "sanna", "Chris Brown",
                                   "Demetri Martin", "Tony Yayo", "melanie martinez", "2 - chainz", "Bryson Tiller",
                                   "Maxwell", "Huey Lewis", "the Rolling Stones", "Bruno mars", "John Mayer",
                                   "Maroon five", "Adele", "Zoolights", "Tori Kelly", "Gabrielle Union",
                                   "The Nutcracker", "Dead and Company", "Ripple", "Nascar", "Rod Stewart",
                                   "Chris Young", "the Rockettes", "R Kelly", "2 Chainz", "p diddy", "green day",
                                   "Jake Owen", "Keyshia Cole", "BRUNO MARS", "Jack Johnson", "Wicked", "12 Steps",
                                   "pride fest", "Steel Panther", "Victoria 's secret", "Blue Man 's group",
                                   "Wheel of Fortune", "Celine Deon", "the Bills", "Fleetwood Mac", "TSO",
                                   "call of duty", "John Legend", "carrie underwood", "Steve Harvey", "Jimmy Lavado",
                                   "Beethoven", "UFC", "AC / DC", "Eric Church", "Hanson", "Jay - Z", "Bruno Mars",
                                   "Lady Gaga", "wotan", "Sam Smith", "coldplay", "The Black Eyed Peas",
                                   "Gabriel Iglesias", "Cold Play", "frank ocean", "lara fabian", "Kid Cudi", "Halsey",
                                   "Pete Yorn", "Pink", "Diana Ross", "the Phoenix Symphony", "daft punk", "Davido",
                                   "mumford and sons", "drake", "Casting Crowns", "Trans Siberian Orchestra", "DNCE",
                                   "Elton John", "Masashi Kishimoto", "John valby", "Lady GaGa", "Shinedown",
                                   "St . Louis chamber orchestra", "Titanic", "Drake", "Kanye", "marvel",
                                   "Black Panther", "Lorde", "pearl jam", "George Straight", "Rhiannon", "greg laswell",
                                   "Rolling Stones", "Jennifer Lopez", "Star Trek", "Depeche Mode", "Jhene Aiko",
                                   "Blake Shelton", "Korn", "Trans Siberia Orchestra", "H.E.R .", "blue mans group",
                                   "Shawn Mendes", "KC Chiefs", "eric church", "the easter bunny", "Riverdance",
                                   "Kelly Rowland", "AA", "bruno mars", "Michael Jackson", "Three Tenor", "Queen",
                                   "lady gaga", "luis miguel", "maroon 5", "Charlie Daniels", "R . Kelly",
                                   "Duran Duran", "Cage the Elephant", "Celtic Woman", "Handel", "Hank Aaron", "Thrice",
                                   "the Misfits", "Charlamagnes", "the Easter Bunny", "Janet Jackson",
                                   "Coastal Cleanup", "Ozzy Osbourne", "Brittney Spears", "metallica", "Pusha T",
                                   "American Idol", "SZA", "Dana Gould", "taylor swift", "tso", "Reds", "Outkast",
                                   "Justin Timberlake", "boyz 2 men", "Three Days Grace", "William Corrigan",
                                   "Josh Groban", "penn and teller 's", "Avenged Sevenfold", "Ciara", "Morrissey",
                                   "santa", "gwen steffani", "Justin Bieber", "snoop dogg", "the Omaha Symphony",
                                   "Kyle Lucas", "Lights Under Louisville", "kanye west", "Florence and the machine",
                                   "the Blue Man Group", "Jlo", "pink", "toys for tots", "J.Cole", "ND Glee Club",
                                   "Baby Metal", "Lindsay Lohan", "Santa", "J . Cole", "Mary Mary", "Dave Matthews",
                                   "eminem", "Miley Cyrus", "Creighton", "R kelly", "Santa Clause", "MLB", "Thor",
                                   "sia", "Hey Ocean", "The Bite of Seattle", "Pentatonix", "Tiwa Savage",
                                   "justin beiber", "red cross", "P ! nk", "Ed Sheeran", "Nine Inch Nails",
                                   "billy joel", "the Wizards", "hinder", "Billy Joel", "mhd", "hall and oates",
                                   "Charlie Wilson", "Pierce the Veil", "Joffrey Ballet", "Bon Jovi",
                                   "Homeowners Association", "Buddy Guy", "Kevin Hart", "jennifer lopez",
                                   "Jason Aldean", "George Strait", "John legend", "sublime", "Katy Perry",
                                   "the Piano Guys", "Mary Kay", "kiss", "Thomas Rhett", "TobyMac", "Jo Dee Messina",
                                   "aerosmith", "The Strokes", "snarky puppy", "katy perry", "Carrie Underwood",
                                   "50 cents", "nin", "Mormon Tabernacle Choir", "Jimmy Buffet", "Kanye West",
                                   "the posies", "Billy Idol", "Anderson . Paak", "JLo", "Coldplay", "LadyGaga",
                                   "Lionel Richie", "the foo fighters", "Tori Amos", "MSU", "Niall Horan",
                                   "Marc Anthony", "The Roots", "Xscape", "Alicia Keys", "Kendrick Lamar",
                                   "springsteen", "Backstreet Boys", "Hillary Clinton", "Disney on Ice",
                                   "St . Louis opera company", "Primus", "noah and the whale", "Luke Bryan",
                                   "Twenty one pilots", "Eagles", "John Mellencamp", "Big K.R.I.T", "santa clause",
                                   "fantasia", "Aerosmith", "Celine Dion", "the lumineers", "brantley gilbert",
                                   "selena gomez", "Phil Collins", "Tylor", "Yo Gabba Gabba", "aretha franklin",
                                   "Rick Springfield", "Paul McCartney", "Mana", "Our Lady Peace", "Jeremy Camp",
                                   "Alan Jackson", "comic con", "cash cash", "Kesha", "Lil Wayne", "Calvin Klein",
                                   "James Spann", "Jagged Edge", "Santa clause", "Lowest of the Low", "Ozzy",
                                   "Dave Chappelle", "ariana grande", "big sean", "Wynonna", "stevie nicks",
                                   "Britney Spears", "the Memphis in May", "Metallica", "Taylor Swift",
                                   "smashing pumpkins", "Rihanna", "Christina Aguelera", "Adel", "bon jovi",
                                   "glass animals", "Octoberfest", "Facebook", "L.L Cool J", "Rene Best",
                                   "Blue Man Group", "Andrea Bocelli", "Avril Lavigne", "Future", "rolling stones",
                                   "kendrick lamar", "Tim McGraw", "wiz khalifa", "garth brooks", "Journey",
                                   "mariah carey", "Theresa Caputo", "Kelly Clarkson", "Chef roble", "Tamar Braxton",
                                   "Jewel", "earth wind and fire", "PJ Masks", "Bill Anderson", "Dierks Bentley",
                                   "Beenie Man", "Jethro Tull", "Foo fighters", "Natalie Grant", "the Nutcracker",
                                   "TobyMac / Brandon Heath", "Panic ! at the Disco", "Elle Hendersen",
                                   "Isabel Allende", "Baltimore Ravens", "Alton Brown", "50 cent", "Demi Lovato",
                                   "David Copperfield", "Justin Biber", "the Davis Elementary Mariachi band",
                                   "dr . Dre", "foo Fighters", "harry styles", "Lauryn Hill", "Chris Rock", "xscape",
                                   "The Offspring", "Bruce Springsteen", "Styx", "PINK", "jay z", "the o'jays",
                                   "Maroon 5", "Handels Messiah", "Santa Claus", "jcpennys",
                                   "National Rifle Association", "impractical jokers", "Five Finger Death Punch",
                                   "Trans - Siberian Orchestra", "Ed Sheran", "Bocelli", "Lucy Liu", "Mariah Carey",
                                   "Tony Robbins", "ACDC", "Keri Hilson", "The Weeknd", "21 Pilot",
                                   "Mannheim steamroller", "Disney", "Daft Punk", "Playstation Experience",
                                   "janet jackson", "Comic - Con", "Godsmack", "the Trans Siberian Orchestra",
                                   "Chris Stapleton", "Martial Canterel", "Rhiana", "Eminem", "Garth brooks", "NKOTB",
                                   "J Lo", "Reba", "Pit Bull", "Fall Out Boy", "Backstreetboys", "U2",
                                   "Tulsa Philharmonic", "Ron White", "dr dre", "Usher", "Foo Fighters", "Jay Z",
                                   "king tut", "keith urban", "tool", "Kenny Chesney", "Jaguars",
                                   "trans Siberian orchestra", "Megacon"]

        return random.choice(reminder_event_name_lst)

    @staticmethod
    def reminder_todo():
        import json
        with open(
                '/home/lzx2000/test/testgit/low_resources_semantic_parsing/generate_dataset/build_labels/instance_funcs/todo.json',
                'r', encoding='UTF-8') as file:
            todo_data = json.load(file)
        return random.choice(todo_data)

    @staticmethod
    def person_name():
        person_reminded = ["condo", "my bowling team", "Shannon", "Brianna", "my", "Phil", "Nour", "flag football team",
                           "the choir", "team members", "Jammin with Bea", "taylors bridal party", "Jade", "kiya",
                           "family", "Chris", "Mauri", "Gus", "Joanne", "Spencer", "Ryan", "coworkers", "be",
                           "the neighbors", "Kharma", "Cousin 's", "mommys", "joe", "Carol", "Owen", "Japan - Taiwan",
                           "the beach club", "Edward", "My", "Tara", "the gardener", "play date", "Sydney", "Jordan",
                           "Amy", "jeff", "Kate", "the soccer team", "Brad", "vendor", "Ray", "PTO", "Isaac", "Paul",
                           "myself", "I", "pep club", "Glenn", "meet up", "Zs", "Bill", "Dylan", "the MC",
                           "the trumpet section", "all senior citizens", "soccer team", "jr", "Alex", "shopping",
                           "Tricia", "Joe", "Jonty", "Cindy", "Team Members", "rich", "car", "reading club",
                           "my mother \u2019s nursing home", "Lucy", "Cycle", "Peaches", "Braydon", "Aimee", "Rodney",
                           "everyone", "Maddie", "Billy", "jacks", "the band", "durby", "Shawn", "Marie", "Steve",
                           "kris", "Zay", "Kellie", "Alesha", "Pixie", "my private students", "Danny", "Bob",
                           "Jennifer", "party planning meeting", "Keith", "Fred", "john", "us", "Jane", "Jim",
                           "Sunday school", "Janie", "everybody", "the board members", "William", "Juan", "tj",
                           "Everett", "jim", "todd", "Frank", "Jenna", "Perin", "James", "Tannis", "Ashley",
                           "gold volleyball", "Xander", "everyone in the building", "jimmy", "Brent", "Cooks", "Dalton",
                           "Brenna", "Nan", "Sherman", "Andrew", "the pool team", "Josiah", "book study", "james",
                           "Colin", "sue", "my painting class", "I've", "Jason", "fitness", "m", "my team", "Jeff",
                           "Erik", "Allie", "Dr . Wright", "Tina", "the team", "Jake", "Bills", "Dawson", "Evan",
                           "groups", "the school board", "the coworkers", "BRIDGETT", "Jody", "dane", "Tristan",
                           "the Fletcher girls", "work", "Quinn", "roxy", "Theresa", "swim club",
                           "the accounting staff", "Joseph", "Mike", "Dixie", "girls", "sophie", "Gavin", "mom",
                           "dance troupe", "the luncheon guests", "Nicole", "Andrea", "the girls", "my class", "Belamy",
                           "lax team", "JD", "alice", "create", "Olga", "Wesley", "Ignite", "Molly", "Joey", "Lori",
                           "everett", "cheerleading", "my teammates", "Dave", "Asia", "Gabriel", "Tom", "study", "myy",
                           "Jivan", "the lawyer", "Teagan", "everyone in the group", "Lucas", "workout", "]", "Lila",
                           "Alyssa", "John", "the caterer", "Otis", "Randy", "The PTA", "Cade", "chef 's table",
                           "charlie", "Barney", "Book Club", "Sam", "the PTO", "Tyler", "Pinky", "Marti", "Grayson",
                           "bowling", "my co - workers", "bob", "JEAN", "hiking club", "Sue", "History", "Derek", "Kay",
                           "soccer players", "Sandra", "the servers", "nascar", "baseball", "Keto", "Vijay", "Julie",
                           "Jay", "team lead", "Gil", "Johnathan 's clients", "Travis",
                           "everyone who 's in the neighborhood", "Tiffany", "church", "Karoline", "Marc", "Family",
                           "Brian", "Antonio", "soccer moms", "dance", "our", "sewing", "Bryce", "bffl", "Katherine",
                           "Arun", "Nathan", "roomate", "Linda", "Marks", "Gary", "Lauryn", "Project 4", "cycling",
                           "sara", "we", "the hockey team", "Dee", "Ashkin", "jerry", "June", "Gibson",
                           "the fitness class", "painting", "Me", "Dunkin", "the group", "ski", "me", "ben",
                           "my book club", "walking", "the office", "workshop", "Koz", "Mason",
                           "the condo association board", "Kevin", "Hank", "bicycle", "book club", "concert buddies",
                           "Sarah", "the Thornbury Software Board", "Leigha", "religious", "Rosie", "Audrey",
                           "the clay guild", "Garth", "Sherry", "Josh", "golf", "Donny", "my coworkers", "Elle",
                           "stefania", "my workout team", "my workers", "steve", "Aidan", "Andy", "Joao",
                           "the choir team", "reading", "MY", "Redhill moms", "the class", "susan", "Camryn", "Sharon",
                           "Rob", "Al", "Sewing", "Marisa", "Carla", "DreAnna", "sales", "Class Reunion planning",
                           "softball moms", "bill", "writing", "Mikayla", "Jacob", "bunco", "PJ", "Gwen", "Lisa",
                           "Jack", "liz", "biking", "Hayden", "Lauren", "Davis", "art camp", "Dean", "pool team",
                           "the consultants", "the GAB families", "Bobby", "ladies night", "Chuck", "Albert", "Belle",
                           "math study", "Benny", "fashion", "Hot Yoga", "the bowling team", "hally",
                           "Highschool Reunion", "Japanese", "Addie", "Carl", "Samantha", "the tutors", "Jude",
                           "each child", "Conlan", "mandy", "Kirsten", "Gerry", "Lindsay", "jogging team",
                           "my cycle group", "Kim", "Isaiah", "the film crew", "my assistant", "single mother",
                           "players", "Groom 's men", "Mo", "BZB", "MY BAND", "Elizabeth", "Lamaze", "Roxy", "Larry",
                           "Joel", "Patty", "Tennis Team", "Percy", "peter", "lisa", "warehouse", "Edd",
                           "the soccer moms", "Brady", "the Bible Study", "Diane", "jeanette", "the girls club",
                           "the girl scouts", "fantasy football", "the whole office", "Allen", "Mrs . Thomas",
                           "the volunteer", "Andi", "Isabelle", "alumni", "Ranu", "all candidates", "Emma",
                           "rottweiler", "Artie", "suzy", "David", "the crew", "Dalena", "Jimmy", "Deardorff Family",
                           "Lytheria", "Shane", "the parents of my music students", "Harmony", "Dennis", "Chances Are",
                           "Alicia", "Jonah", "DoI", "Dustin", "PTA", "Audreys softball team", "the babysitter", "Cam",
                           "the parents", "Benji", "cooking", "neal", "Brandon", "marshalls",
                           "the Rutledge Lake employees", "the girls in the dance troupe", "Deb", "Nide", "Cody",
                           "George", "I'm", "Sally", "Coty", "Anna", "weight loss", "couples golf league", "Capri",
                           "chris", "stephanie", "Luigi", "cade", "Eli", "the horse club", "Theater", "camping", "Zach",
                           "Walker", "football moms", "the scouts", "the Street Crew", "cooking class", "Mark", "Ben",
                           "Matt", "sword club", "Aiden", "Preschool", "Krissy", "the Liz", "ME", "Dagan",
                           "the network team", "jason", "Robert", "Carly", "card club", "girls trip", "office people",
                           "Delali", "volunteer", "HR", "the flag football team", "the Board of Thornbury Software LLC",
                           "Tony", "the bookclub", "jessica", "Serenity", "the swim team", "the baseball team", "i",
                           "the running club", "the book club", "Cary", "Jen", "Caitlyn", "Anita", "Jessie", "Gomez",
                           "Popcorn Friday", "Jeri", "Dan", "Magda", "Stephan", "running", "Team Leaders", "Addison",
                           "jack", "the family", "Garden", "Zumba", "Mary", "Q team", "Taylor", "boy scout troop",
                           "the Council Trainers", "the students", "Katie", "Susan", "JULIE", "bart", "art", "Liz",
                           "i've", "mother 's morning out", "Cathy", "him", "the cleaning lady", "Nick", "stephen",
                           "golfing", "Crystal", "Tammy", "Lee", "Carrie", "TS", "Norma", "employees", "tyler", "Megan",
                           "the maid", "Loren", "Special needs mom", "card", "Arabella", "Paulie", "Tim", "moms", "I'd",
                           "park", "Dr harris", "em", "marv", "Maura", "Janice", "Darcy", "Laundry", "art club",
                           "staff members", "Xavier", "the boys", "tom", "aa", "youth", "madelyn", "the staff",
                           "Joshua", "dart team", "boys", "the interview team", "Gertha", "group", "Ally", "emily",
                           "Jonathan", "Rebecca", "jake", "hiking", "SCA members", "baby wearing", "music", "Ghost",
                           "Emerie", "the running group", "Alisha", "SAMANTHA", "radiology", "Emily",
                           "my spinning class", "Brittany", "Lea", "team", "carol", "my personal trainer", "food",
                           "Jing", "Phillip", "Will", "Bonnie", "Amanda", "Karly", "friends", "andy",
                           "everyone but steve", "Alan", "Stella", "Stewart", "Amory", "rex"]
        return random.choice(person_reminded)

    @staticmethod
    def category_event():
        category_event = ["lights competitions", "Wine and chocolate events", "Firework displays", "skatepark events",
                          "special program", "songwriters festival", "lighting festivals", "bowling", "Meetups",
                          "Happy hours", "music related events", "Flea Markets", "the lighting ceremony",
                          "Comedy shows", "sports car shows", "the food drive event", "yards sales", "mass",
                          "Fireworks Displays", "carnival", "Hiking meetups", "symphony orchestra performance", "SKI",
                          "Sleigh Rides", "food sampling", "the art fair", "opera", "volunteer", "drinking events",
                          "musical plays", "horse back riding", "tattoo conventions", "back to school parties",
                          "sledding", "Chippendales", "trivia nights", "Monster truck races", "rodeo",
                          "Star Wars launch events", "beer festivals", "Tree walk", "Movie marathons", "Pop up shops",
                          "music festival", "flower and garden festival", "live music events", "ballets",
                          "Broadway shows", "the big 500 parties", "cross country skiing", "Oktoberfest events",
                          "zip lining", "Gingerbread House decorating parties", "plane show", "Sample sales",
                          "reggae concert", "Light display", "any nature shows", "Apple picking activities",
                          "a book fair", "home and garden shows", "Movie", "Lights show", "beach party",
                          "Holiday Events", "ornament parties", "painting classes", "Chowder festivals",
                          "food festival", "anime con", "religious services", "window displays", "movies", "TED Talks",
                          "Flea market", "the Pretty Princess Party", "the Blues Festival", "night skiing",
                          "Anime Conventions", "tree lighthing events", "craft beer festivals", "classical musicals",
                          "a yard sale", "Santa Con", "comedy shows", "pick apples", "mommy and me yoga",
                          "Hotel events", "CONCERTS", "craft beer meetups", "sport comedy event", "dining",
                          "Cake decorating classes", "miss texas", "tastings", "music shows", "design classes",
                          "spinning classes", "Baking parties", "live reggae", "maple syrup festivals",
                          "Tailgating events", "professional fights", "parades", "The lighting ceremonies",
                          "charity benefits", "metal concerts", "THE WINE FESTIVAL", "BASKETBALL GAME",
                          "wine and cheese", "Live concerts", "Literary conventions", "Broadway Shows",
                          "the Nutcracker", "the downtown Christmas lights", "the music concert", "festive",
                          "electro party", "Pen pal events", "Gospel Concert", "wine tasting events",
                          "book reading event", "a light parade", "Exhibit openings", "Food market",
                          "Kids clothing sale", "celebrations", "the neo - soul happy hour", "Pictures",
                          "Bluegrass festivals", "Adult night", "Grand store openings", "Soccer clinics", "Ballet show",
                          "celebration events", "bale performance", "birthday events", "festival", "art walks",
                          "Art Exhibit", "Comedians", "movie premiers", "art festivals", "carriage rides",
                          "live music playing", "festival events", "Festival events", "political events", "lineup",
                          "Art shows", "Crafts shows", "Parades", "Night Market", "Gaming expo", "sing",
                          "brewery tours", "light display", "fun run", "art shows", "Sport events", "Brunches",
                          "blues concerts", "Harvest festival", "the dinner", "Comedy festivals", "rap concert",
                          "Shakespeare events", "Musical performances", "fitness events", "movie screenings",
                          "seasonal concerts", "performs", "a snowmobile guided tour", "Beer or wine tastings",
                          "Magic shows", "Art Gallery openings", "parties", "sports", "a wine festival",
                          "classical performances", "Dance competitions", "Nutcracker performance",
                          "Face panting party", "Ice shows", "5K races", "the All Souls Procession", "pumpkin patch",
                          "online sales", "Christmas tree cutting", "the country music festival", "concert series",
                          "Pumpkin carving parties", "costume parties", "the parade", "movie night",
                          "1980s cover bands playing", "country bands", "murder mystery dinner theatre productions",
                          "wine tours", "beer or wine walks", "Pampered Chef parties", "beer tastings",
                          "Tree lightings", "Any live music", "vendor shows", "the annual running races",
                          "art workshops", "the swap meet", "christmas lights shows", "the featured performer",
                          "Meeting", "a costume contest", "fiesta latina", "yoga classes", "Christmas in the Park",
                          "benefit concert", "the food for the homeless sharing event", "classical music concerts",
                          "Erotica Reading meetups", "Open Mic", "ice carving events", "golden globe awards",
                          "comic con", "cultural events", "podcast events", "Clay molding parties",
                          "candlelight services", "a Secret Santa drive", "flashmobs", "beer events",
                          "a Hearthstone Fireside Gathering", "Singles meetups", "Messiah show", "Star wars movie",
                          "Dance classes", "musical concert", "musical theater", "breast cancer walk", "arts festival",
                          "beer festival", "high school plays", "community theatre events", "light displays",
                          "a holiday play", "Drone Racing Courses", "showing the Pink Panther movie", "boys night out",
                          "live jazz", "Toy drive", "street festivals", "tasty food events", "food trucks",
                          "reservations", "Gaming Cookouts", "seafood and wine festivals", "art openings",
                          "MUSIC CONCERTS", "Coffee hour", "food & wine", "holiday stuff", "a salsa class",
                          "a haunted house", "magic shows", "the ice skating", "Body painting",
                          "Livestock Show and Rodeo cookoff", "exhibit", "market", "Viking things", "surfing event",
                          "Music", "food", "boat parties", "ballet performances", "Meditation classes",
                          "charity parties", "Truffle making classes", "yoga retreat", "Cheese and soap making classes",
                          "football parties", "Dance shows", "Chess tournaments", "DairyFest", "Flea markets",
                          "fashion show", "skating", "Walking Tours", "Ice fishing contests", "Tamale festivals",
                          "Super bowl events", "jazz festivals", "the festival", "THE BAND", "the baking party",
                          "glass blowing shows", "laugh", "the marathon", "Hip hop concerts", "Nutcracker",
                          "science fairs", "Craft fairs", "the holiday parade", "cookie bake sales",
                          "the wine festival", "an egg dying party", "Wine tasting parties", "game plays",
                          "acoustic performances", "local bands playing", "Brew Fests", "hockey game", "the Air show",
                          "Broadway show", "Snowboarding events", "Happy Hours", "jazz concerts", "a Broadway show",
                          "Dinner party", "Holiday movie nights", "Festival of Carols Concert", "Vintage Car Show",
                          "car shows", "the gingerbread house contest judging", "Science events", "wine - tasting",
                          "reading events", "the food festival", "alternative bands playing", "the Polar Express",
                          "fairs", "cover band", "cookie parties", "Live plays", "Hip hop night", "pub crawl events",
                          "Exhibit", "Church programs", "the concert", "Law enforcement charity benefits", "basketball",
                          "paint party", "Music Shows", "comic book night", "the light fest", "Pumpkin picking",
                          "Caroling", "Pumpkin festivals", "Festival of the Lights", "literary events",
                          "Rooftop parties", "blood pressure screening", "strawberry picking activities",
                          "tailgating events", "Gospel Music lunches", "Craft festivals", "beer and bacon tasting",
                          "Holiday pop up market", "the Nutcracker play", "Drag shows", "a corn maze", "School events",
                          "Professional bull riders events", "country music singers", "the artist reception",
                          "Wine social events", "a concert", "Skating events", "ballet performance", "games",
                          "pet adoption events", "Knitting circles", "sports events", "cookie baking places", "Parade",
                          "Beatles tribute band", "BEACH PARTIES", "Movie premiers", "perform", "performing",
                          "cuban music", "cooking workshops", "Fall Festivals", "ice arena", "holiday shopping events",
                          "masquerade", "a game", "the music expo", "Rivers of Light", "the taste of Chicago",
                          "Jazz music", "a rap concert", "Book clubs", "the circus", "holiday event", "Car shows",
                          "live reggae shows", "a music concert", "Ice skating classes", "brunch specials", "Karaoke",
                          "treelighting events", "Christmas Houses", "The Nutcracker Ballet", "volunteer opportunities",
                          "Bowling tournaments events", "the Bass Cat tournament", "the total eclipse", "Beer Festival",
                          "Country line dancing", "Pizza festival", "a show", "sporting events", "concert presales",
                          "a live country band", "charity runs", "Performances of the Messiah", "the pottery classes",
                          "Live indie bands", "the Winter Wonderland", "Activities", "Beauty & the Beast play",
                          "hip hip DJ", "Pop artists", "masses", "Haunted house tours", "live music shows",
                          "count down party", "fashion events", "Body Exhibit", "holiday food events", "local music",
                          "Networking dinners", "book fair", "the master garden show", "hot yoga classes",
                          "a bike race", "Pierce County Fair", "new years party", "Yoga classes",
                          "Video game tournaments", "Happy hour", "pottery events", "production", "the Ghost Tours",
                          "village display", "musical performances", "Holiday activities", "community yard sale",
                          "Comedy Shows", "the tree lighting", "wine and paint nights", "E3", "rap concerts",
                          "a Jazz festival", "the world tournament for league of legends", "pumpkin carvings",
                          "Hip hop parties", "Food deals", "Food truck rally", "Paint Nite parties",
                          "tree decorating parties", "Peace protests", "Grims Fall fest", "Film festivals",
                          "introducing their new giraffe", "the show", "a birthday party", "Train show",
                          "the art festival", "Photography workshops", "walk", "a rock concert", "pop up restaurants",
                          "Movie festivals", "school", "the music and wine festival", "HOLIDAY FESTIVITIES",
                          "Bethlehem", "Comic Conventions", "ceramics workshop", "food drives",
                          "singer songwriter events", "Boxing", "a winter festival", "trunk shows",
                          "beauty and beast play", "Musicology seminars", "Reptile show", "beer tasting", "live shows",
                          "Cheese and Wine tasting parties", "monster trucks", "Beer fest",
                          "gingerbread decorating parties", "Derby", "caroling events", "opening acts",
                          "a sporting event", "the music festivals", "pop up store event", "Book club events",
                          "the taekwondo tournament", "EDM show", "choir performance", "Sensory day", "car show",
                          "Dinner", "rock music", "Snow tubing", "the Hot Air Balloon Show", "music", "Art walks",
                          "the Greek festival", "roof top concert", "watch party", "Auto Show exhibit", "a happy hour",
                          "Beach Party", "wine tasting", "pumpkin picking", "fall festivals", "Hot dog eating contests",
                          "Swim classes", "the town meeting", "a football game", "punk bands playing",
                          "Bottomless mimosa brunches", "Photo shoots", "Flower show", "Ice Skating",
                          "fireworks displays", "the cantata", "bar crawls", "horse drawn carriage rides", "Live music",
                          "Organized running races", "Dog festivals", "Poetry readings", "poker tournament",
                          "art events", "craft show", "Antique shows", "Walking tours", "Football viewing parties",
                          "the ballet", "Breakfast", "polar express", "the Pirates", "health screenings", "cruises",
                          "the jazz festival", "BDSM events", "Film showings", "annual events", "choral concerts",
                          "night life", "church services", "lights display", "a nutcracker performance", "opening day",
                          "Cookouts", "nature walks", "the Parade", "lunar eclipse", "Harvest Festival",
                          "bonfire parties", "a car show", "ice skating events", "Football games", "carolling event",
                          "the wine and food festival", "food truck events", "fireworks show", "broadway plays playing",
                          "Easter egg hunt", "open mic shows", "Tupperware Parties", "ethnic events",
                          "tree lighting 's", "country concert", "seasonal events", "the Wyoming Co . Fair",
                          "job fairs", "soap making classes", "theatre events", "crafting events", "Dance parties",
                          "potluck", "sale events", "Festivals", "THE MOVIE", "Vocal classes",
                          "Tree lighting ceremonies", "Disney on Ice", "any art classes", "Any sporting events",
                          "Ugly sweater parties", "the Ferrari Festival", "Book signings", "Beer festivals", "bingo",
                          "Improv shows", "the diabetic walking", "meetings", "Mardi Gras activities",
                          "star wars showing", "baseball", "Ladies night", "an art gallery showing",
                          "ugly sweater parties", "election", "rodeos", "Musical Theater", "race events", "art",
                          "park festivals", "holiday happenings", "Music event", "restaurant events",
                          "chamber groups playing", "getaway", "Italian food festival", "haunted houses",
                          "Speed dating", "the Opera", "artists", "light viewings", "orchestra events", "reading",
                          "Pie baking contest", "Motorcycle rallies", "the main comedy show", "tours",
                          "Swimming lessons", "Cocktail related events", "Jazz shows", "broadway",
                          "Cooking decorating events", "Passion Parties", "Condo events", "Painting parties",
                          "Garlic festival", "Burlesque shows", "hip hop shows", "Off Broadway shows",
                          "a walk - a - thon", "Live music events", "country concerts", "show", "Rodeos",
                          "Biking events", "the Pride Parade", "ladies night", "food drive", "Musical premiers",
                          "Musicals", "cande lit virgil for the victims", "Pumpkin patches", "Live acoustic music",
                          "Christian Music concerts", "hiking", "Meet and greet", "the tour of lights", "Holiday plays",
                          "Food tasting events", "theater performance", "Food trucks", "Sporting events", "Craft shows",
                          "wine tastings", "wine taste parties", "Food events", "Circus events", "craft fairs",
                          "job fair events", "Disney on Ice events", "Parties", "date night", "tree decorating events",
                          "Support groups for grief", "store visits", "presidential election",
                          "the California International Marathon", "the Wine Tasting", "film festivals", "Foam parties",
                          "Date night", "Concert events", "holiday lights festivals", "Movie production",
                          "punk concerts", "a toga party", "candlelight historical home tours", "Music festivals",
                          "cake parties", "a comedian", "workshops", "Chocolate festivals",
                          "the Hudson Valley wine tour", "hip hop concerts", "Renaissance events", "the school concert",
                          "Pottery Classes", "live nativity performances", "gaming tournaments",
                          "the My Favorite Murder live podcasts", "painting events", "Nutcracker shows",
                          "tree light ups", "Tour", "firework shows", "hair shows", "Microfestivus", "move",
                          "a piano concert", "pics", "any musical plays", "Fashion shows", "Christmas movies",
                          "movies premier", "professional sporting events", "Musicals playing", "brunch events",
                          "Winter Olympics", "jazz concert", "Open mic nights", "E - Sports Related", "Gun shows",
                          "Anime expo", "music event", "meditation classes", "Christmas Carol", "Cultural events",
                          "the holiday concerts", "the battle of the bands", "Volunteer events", "toys for tots drives",
                          "The Last Jedi", "opera events", "book signing", "a lighting ceremony",
                          "opera caruso performance", "Movie showings", "puppet shows", "the Classical music show",
                          "live stand up comedy", "a Hairstyles for Long Hair workshop", "the news", "sporting event",
                          "Pagan parties", "Tractor pull contests", "Haunted houses", "Festival of Carols", "ballet",
                          "wine and food events", "art classes", "Conventions", "the state fair",
                          "the festival of lights", "church plays", "the tree lighting ceremony",
                          "Cookie decorating classes", "Brewery tours", "Movie premiere events", "christmas parties",
                          "music playing", "music concerts", "the flashback concert", "vegan food events",
                          "Step - dancing competitions", "Sports events", "party events", "Beer events", "Poetry night",
                          "fire cracker show", "Holiday parties", "a classical guitar concert", "tour", "motorcycle",
                          "tech related events", "cookie decorating parties", "broadway plays", "bands",
                          "cookie decorating", "the nightlife", "beer and meat events", "rally",
                          "the Wild Lights shows", "Jewelry - making workshops", "carols", "paying", "Mermaids",
                          "Apple picking", "live performances", "Poetry reading", "ice skate", "day parties", "Ballet",
                          "Horse Racing", "giving samples", "celebrate", "the winter Wonderland Craft Event",
                          "DIY classes", "booze cruise", "Festivities", "Gamestop Expo", "Holiday Fairs events",
                          "band playing", "marches", "game", "activity", "cooking demonstrations", "the paint parties",
                          "Live Music shows", "Food Truck Rally", "Expo", "performing the Nutcracker",
                          "party painting classes", "Cheese and wine tasting", "Paint parties", "action movies",
                          "farmer 's market", "Cruisin ' OC show", "concealed weapons or firearms training classes",
                          "a spring festival", "Broadway plays", "noelle", "holiday wine tasting event",
                          "classical music", "a tree lighting", "Live sports", "pop up store", "Neon night bowling",
                          "Girl 's night out events", "circuses", "Cooking Shows", "Riverwalk events",
                          "community projects", "Comedies", "an open house", "storytime", "Boat Parade",
                          "the food and wine festival", "the champagne celebration", "family get together", "Goat Fest",
                          "krampus haunted house", "tapings", "the eclipse", "marathon events", "Fireworks show",
                          "living nativities", "Potluck", "the night life", "movie", "Concerts", "Overwatch tournament",
                          "the meetup groups", "bake sale fundraisers", "exposition", "auctions", "jazz jam sessions",
                          "dog show", "restaurant week", "Holiday concerts", "Networking parties", "an Open Mic event",
                          "the BBQ brunch", "winterfest", "Rodeo", "house party", "Karaoke parties", "costume contests",
                          "concerts", "holiday music concerts", "football games", "classes on investing", "pictures",
                          "Brew festivals", "Ice cream parties", "Poetry reading events", "gingerbread making parties",
                          "Polar Express viewing party", "Tuber ware parties", "Vap parties", "Paw Patrol Live",
                          "the book signing", "BBQ events", "tamale festival", "shows playing", "anime conventions",
                          "Cookie tasting", "Political Rallies", "church", "Fly fishing competition",
                          "Electronica rave music", "a dog show", "food events", "display", "cinema", "street fair",
                          "skating events", "school events", "apple picking", "art expositions", "pumpkin patches",
                          "fair", "wine testing events", "line dancing", "boat show", "art showings",
                          "Hair show events", "the Renaissance Fair", "live concerts", "a book signing", "Wine tasting",
                          "visits", "Dog costume contests", "grilling contest", "hip hop show", "decorate cookies",
                          "Dog walking group events", "Whale watch", "a painting party", "food tours",
                          "middle school band concert", "an eating contest", "Community events", "the wine festivals",
                          "the Super Bowl", "comedy", "Paintball competitions", "playing", "arts and crafts events",
                          "live theatre shows", "strawberry picking events", "christmas tree lightings", "Showings",
                          "arts & crafts fairs", "Birthday celebrations", "dance events", "the street festival", "wine",
                          "the Nutcracker performance", "ferry rides", "Sale events", "farmer 's markets",
                          "kosher dinner", "gaming events", "Chocolate Festival", "Tours", "PARADE", "hip hop events",
                          "water events", "wine and cheese tasting", "Sesame street live show", "Christmas lights",
                          "community garage sales", "Pumpkin Carving parties", "musical concerts", "Wine Festival",
                          "happy hour specials", "Event openings", "Dine in la", "films", "Football parties", "Pride",
                          "Ice cream socials", "lights shows", "Wine Tastings", "musical", "Kite festivals",
                          "Food eating contests", "tree lighting ceremonies", "the Clintonville bar hop",
                          "Paint nights", "blood drive", "dog walks", "Music release parties", "the music events",
                          "Jazz", "Warm clothing drive", "club events", "Movie premier", "Art gallery openings",
                          "rap shows", "yoga events", "Comedy events", "bike rides", "Charity events", "state fair",
                          "Scrapbooking parties", "the lighting ceremonies", "wine testing", "the party", "sales",
                          "meeting and greeting event", "paw patrol live", "latin dancing", "Christian singles events",
                          "Pumpkin festival", "E - Sports Parties", "dog training classes", "surfing competitions",
                          "blue concerts", "bands playing", "The vintage handmade market", "lights",
                          "the Seeing Red party", "Kids Fest", "iceskating", "corn mazes", "toys for tots event",
                          "errands", "Movies", "an ice cream festival", "bar crawl", "ukulele concerts", "DJs",
                          "Hay rides", "Luau", "festivities", "5k", "food tasting events", "Gift wrapping parties",
                          "greeting and meeting", "Succulent pumpkin making classes", "the 20 year reunion",
                          "breakfast", "a birthday party event", "star wars events", "Live theater",
                          "Fishing tournaments", "Live Blues", "an art gallery show", "Swimming meets",
                          "Gingerbread house events", "Self asteem seminars", "the grand grand opening",
                          "Theater events", "light show", "playing a show", "Fitness classes", "Hip Hop shows",
                          "workshop", "Wine and food tasting", "grand opening for Toy Story Land",
                          "Gingerbread decorating contests", "musical events", "the costume party", "Painting events",
                          "Farmers markets", "Science fiction conventions", "Boat parade", "Comedy club shows",
                          "Roosterteeth Expo", "Theater productions", "conventions", "Schoolhouse Rock",
                          "the food truck festival", "cider tasting", "football game", "Group runs", "Half marathons",
                          "Country music events", "RWA convention", "Running Events", "Used car auction",
                          "cooking competitions", "tweet events", "live Nativity scenes", "an opera performance",
                          "Hayrides", "photos", "performance art", "caroling performances", "Spanish concerts",
                          "classical concerts", "Comic - Con convention", "the New Beginnings Horse show",
                          "Christmas caroling", "Farmers Markets", "MMA Events", "Food truck events", "Crafts fair",
                          "Animal rights demonstrations", "the spring fling", "a light show", "pool parties",
                          "a live nativity", "Junior night party", "the awards ceremony", "fitexpo",
                          "wine or beer tasting events", "SEWEE", "PAC events", "H.P . Lovecraft fan meeting",
                          "Vinyl music hall events", "meteor shower", "alternative music concerts", "ballet show",
                          "the fall festivals", "the lighting of the Christmas tree", "Beers on the Beach Festival",
                          "esports events", "Open Mic nights", "Bowling parties", "marathons", "country music concert",
                          "hot dog eating contest", "any carnivals", "a cruise", "Yard sales", "painting parties",
                          "ball dance", "the holiday sale", "the fair and rodeo", "activites", "Wine events",
                          "tournaments", "Outdoor movies", "Book club", "water activities", "the Lion King opening",
                          "an improv show", "comic events", "music artists", "light shows", "Ugly Sweater parties",
                          "Wine parties", "roller skating", "Apple pickings events", "farmers markets", "Soccer games",
                          "Food Festivals", "dog adoption event", "the wing and beer Festival", "food festivals",
                          "back to school", "convention", "board game meet ups", "walking tours", "the 5k races",
                          "Pet pictures", "Secret Santa parties", "Drag Shows", "band practice", "Black tied parties",
                          "blizz con event", "ice skating parties", "the lion king", "the boat parade",
                          "Christmas tree lighting", "performance", "poetry reading events", "jazz shows",
                          "restaurant grand openings", "mega concert", "boat Parade", "cookies", "Chinese ballet shows",
                          "new movies", "marathon", "music bands", "support groups", "Golf tournaments",
                          "Tupperware parties", "wine tasting event", "wine - tasting festival", "adult entertainment",
                          "picture taking", "musicals", "the Olympics", "storytelling", "a parade", "wedding expo",
                          "Music Events", "the wine tasting event", "Holiday light displays", "any tournaments",
                          "a comicon event", "Culture", "sport", "bachelorette party shows", "early mass", "Air shows",
                          "job fair event", "Ugly Christmas Sweater parties", "Beer tastings", "shopping events",
                          "baseball games", "outdoor concerts", "opera concerts", "tree lightings",
                          "Super Bowl parties", "light parade", "comedy show", "crab feeds", "lades night",
                          "an 80's cover band", "a music event", "Theater performances", "Dance events",
                          "Rave dance events", "Band performances", "face painting", "Wine and cheese tasting events",
                          "Hay ride events", "Holiday Market events", "Cosplay events", "Hip Hop music festivals",
                          "Product launches", "pottery painting parties", "photo booths", "EDM shows",
                          "Oktoberfest parties", "Drone Racing Events", "the Crazy Car Art Parade",
                          "Cat adoption events", "tree lighting parties", "after - hours parties", "household auctions",
                          "Carnaval", "cooking classes", "food event", "sports related", "train rides",
                          "the songwriter festival", "RV show", "pumpkin maze events", "the fireworks show",
                          "Any restaurant grand openings", "music concern", "Wine festivals", "Turkish festival",
                          "marathon event", "dinner parties", "Street musicians", "Face painting", "Dinner cruises",
                          "self defense classes", "Wiccan", "pick blueberries", "Journey to Bethlehem",
                          "Wine taste testing events", "Farmer 's markets", "Toys for Tots drives", "aviation events",
                          "a food truck convention", "5k races", "costume contest", "Tennis matches", "taco tuesday",
                          "watch whales", "Pool parties", "French Bulldog events", "Film premieres", "fashion shows",
                          "Movie night", "the community Christmas Caroling", "Star Wars", "any opening",
                          "meet and greets", "Gaming events", "Holiday parade event", "the christmas light tours",
                          "dinner", "a first aid class", "Football events", "Restaurant week", "r & b music events",
                          "drag shows", "Couponing classes", "Graduation celebrations", "doctors appointments",
                          "dinner events", "5ks", "food and wine tastings", "the Renaissance Festival",
                          "Carriage rides", "Tree lighting", "entertainment", "A CONCERT", "the jazz performance",
                          "High school football events", "the Nutcraker", "bodybuilding competitions", "a movie",
                          "Brunch events", "new bands playing", "baseball game", "Medieval festivals", "ice - skating",
                          "Restaurant openings", "operas", "Dancing", "baking parties", "pizza making party",
                          "Egg hunts", "medical appointment", "Food pop ups", "Grief support group meetings",
                          "girls night out", "tree cutting", "wine glass painting events", "wine or food events",
                          "writing workshops", "circus", "Art Exhibitions", "Nutcracker Ballet",
                          "Volunteering opportunities", "Doggie 5K", "orchestras", "Comic conventions", "Festival",
                          "draft event", "Cook out", "fighting", "celebration", "comedy club shows",
                          "Gingerbread house building competition", "Listening party", "any open mic nights",
                          "Attractions", "open mic", "Performances", "Dance concerts", "the open house", "picture",
                          "Country Music concerts", "shindigs", "music events", "the triathlon", "happy hour",
                          "release parties for the new game", "sporting", "Funny acts", "Country music concerts",
                          "tree lighting event", "Country music festivals", "parade", "brunch", "Theater plays",
                          "meeting", "pub crawls", "after game parties", "trunk or treat", "video game events",
                          "off - broadway shows playing", "chili cookoff", "Classical music shows", "museum exhibits",
                          "Nutcracker performances", "Renassaince fairs", "Football watch parties", "book readings",
                          "a Tupperware party", "concert", "5K runs", "Jazz Festival", "Art exhibits",
                          "cookie decorating events", "masquerade balls", "craft classes", "karate classes",
                          "Feed the homeless events", "public skating", "CD release party", "Comedians performing",
                          "Theatre performances", "the Nutcracker ballet", "Costume contests", "Craft Fairs",
                          "lights festival", "a wine tasting event", "Cars 3", "25 year high school reunion",
                          "the Chinese Festival", "a christmas cookie decorating party", "dog shows",
                          "Walker Stalker zombie convention", "Rocky Horror", "sports related things",
                          "lighting ceremonies", "Any wine tastings", "happy hours", "the school book fair event",
                          "boating safety classes", "holiday parties", "birthday party", "charity events",
                          "Trivia night events", "church events", "Beer festival", "food sampling events",
                          "fundraisers", "Wood crafting parties", "Dog meet ups", "hot air balloon rides",
                          "dinner outreaches", "comic performing", "tree lighting", "west coast rappers",
                          "wine festivals", "Foodie events", "Gaming Conventions", "a grunge band playing",
                          "the rock and geology exhibit", "Thor movie", "Festival of Lights", "alternative bands",
                          "superheroes", "boat shows", "a horse show", "celebrity DJs", "Theater acts",
                          "Toy drive events", "Craft events", "Bingo", "light", "Musical events", "charity", "Carnival",
                          "an Art Festival", "touring", "soccer hiring events", "Adult entertainment", "live blues",
                          "Free skating", "annual sale", "Mass", "Concert", "DIY workshops",
                          "Openings of new restaurants", "dog surfing events", "a cookie making event",
                          "panting parties", "donuts", "Hay Rides", "Book readings", "Film events", "holiday markets",
                          "the gingerbread house decorating party", "Coachellafest", "musical show", "poetry readings",
                          "educational events", "pumpkin contests", "elderly party", "appearances", "Springfest",
                          "the ACL festival", "ice skating rinks", "paw patrol live event", "horseback riding",
                          "Wine tastings", "service", "Chalk the Block", "Equestrian show", "live wrestling shows",
                          "Deep sea fishing", "Cocktail carving parties", "oscars party", "Disney on Ice shows",
                          "singles nightlife", "civil war re - enactments", "The NUtcracker", "Bazaars",
                          "the extra magic hour", "Baking classes", "Summer Solstice celebration", "learn paddling",
                          "Sporting", "story time", "artists are playing", "programs", "food tasting", "the boat show",
                          "trolly tours", "continuing education classes", "skating event", "an opera",
                          "Vegetarian events", "Oktoberfest", "ballroom dance", "musical festivals",
                          "Pictures with Easter Bunny", "wine fest", "Wine and paint events", "Pumpkin carving events",
                          "folk music concerts", "Volunteer work opportunities", "Any wine festivals", "car racing",
                          "dance party", "glow bowling", "Nightlife events", "wine trail", "Symphony performances",
                          "romantic getaway", "Paint Night parties", "Jazz festivals", "rave parties",
                          "the Rock and Roll Hall of Fame ceremony", "comedians", "surfing", "holiday light displays",
                          "Pumpkin carving contests", "any parades", "ugly christmas sweater parties",
                          "concert for cancer for children", "Bowie Is exhibit", "pet adoption event",
                          "pop music concerts", "Wine and music festivals", "nightlife", "live music show",
                          "the beach party", "lions daze", "Christian Concerts", "Dog training classes", "groups",
                          "Hair cut appointment", "exhibits", "Museum shows", "carriage ride", "5K event",
                          "live comedy shows", "pride", "Comic shows", "carol concerts", "OVOfest", "a date", "Ozzfest",
                          "party", "Honor Flight", "Cookie decorating contests", "a painting event", "speed dating",
                          "flea markets", "phantom of the opera playing", "playing a concert", "monster truck events",
                          "Jazz concerts", "cat shows", "breast cancer awareness", "comedy events", "Nutcracker ballet",
                          "the music group", "concert shows", "Kite - flying contests", "the happy hours", "bon fires",
                          "HIP HOP concerts", "a healthy lifestyle", "the music event", "artsy event",
                          "Big Band concerts", "the Bingo Match came", "Fairs", "casting calls", "bake sales",
                          "the monster truck rally", "Creative writing workshops", "tailgate parties", "Plays",
                          "Facebook events", "county fairs", "Pub crawl", "alumni ball",
                          "the showing of the Tulip Festival", "book clubs", "Ski lessons", "Broadway",
                          "Breakfast with Santa events", "car auctions", "a party", "Food festivals", "sleigh riding",
                          "yard sales", "Paint nite events", "Costume parties", "Musical concerts", "Rap concerts",
                          "Clay sculpting parties", "festival of lights parties", "dancing", "homesteading expo",
                          "Acting workshops", "Kpop concerts", "Craft Beer events", "LGBTQ events", "Brunch",
                          "Art events", "Street fairs", "Cheesecake festivals", "Birthday event", "pictures with Santa",
                          "ballet shows", "Farmer market events", "college events", "Autism Awareness events",
                          "holiday events", "the concerts", "Rock concerts", "Sports", "restaurant openings", "Skiing",
                          "hayrides", "Food tours", "the beer events", "a comedy show", "musical entertainment",
                          "broadway shows", "the Christmas Tree lighting", "stand - ups", "Taco events",
                          "standup comics", "Music played", "gallery", "Art Shows", "Story time", "rock climbing",
                          "Disney On Ice", "markets", "quilting parties", "cocktail party", "Costume party",
                          "Classic car shows", "a golf match", "dinners", "sewing classes", "sample sale",
                          "the Pumpkin Blaze", "the lighted boat parade", "Miata meetings", "Balloon Fiesta",
                          "Wine tours", "Paint Nite", "Pumpkin carving", "premiere", "dance club openings",
                          "Tree decoration", "the harvest festival", "the air show", "open mic night",
                          "the birthday party", "All - you - can - eat - contests", "movies on planes",
                          "nature activities", "boat rides", "Wreath making classes", "disney on ice shows",
                          "tours of popular pop singers", "paint nite", "Romeo and Juliet", "Sales", "Cocktail parties",
                          "cooking class", "Strip dancing workout classes", "Hiking group events",
                          "Night life highlights", "Food event", "plays", "live Ice performance", "Wine tasting events",
                          "music festivals", "the Big Fishing competition", "Christmas Tree decorating party",
                          "the cars race", "holiday boat show", "a cancer walk", "us president election", "folk music",
                          "Home Show", "Caroling events", "HOLIDAY PARADE", "farmers market", "the Winter Blast",
                          "Wine tasting activities", "garage sales", "Publicity concerts", "Zoo events", "sport events",
                          "Ballets", "horror conventions", "Vegan festivals", "paint nights", "Book festivals",
                          "Craft Fair Events", "the Festival of Lights", "a Happy Hour", "sweater contests",
                          "Art festivals", "Anime events", "a Town Wide Yard Sales Event", "pottery classes",
                          "excursions", "Hip - hop festivals", "slam poetry", "Film screenings", "Business conferences",
                          "entertaining", "Classical concerts", "Comedy Events", "Comedy competitions",
                          "church concerts", "taking pictures", "musical theater events", "the haunted houses", "SXSW",
                          "a karaoke night", "STEM events", "Fiesta", "a play", "dj sets", "nature themed happenings",
                          "dentist appointment", "5k runs", "sales events", "Happy hour deals", "a wrestling match",
                          "Classical Concerts", "volunteer serving food", "music Festivals", "air show",
                          "Bridal events", "ice skating", "metal bands", "Indie concerts", "Music festival",
                          "country music concerts", "the comedian", "jazz music", "back to school events",
                          "Pagan potluck", "board game night", "Science Fiction conventions", "Speaking events",
                          "Clothing sales", "estate auctions", "Photography classes", "Opening Day parade",
                          "Movie screenings", "the special exhibits", "Renaissance fairs", "soccer games",
                          "holiday fairs", "Infinity War", "the live music", "annual yard sale", "romantic events",
                          "barbeque festival", "TREE LIGHTING EVENTS", "live music performances", "acrobat shows",
                          "art exhibits", "the Star Wars movie release", "Wine tasting festival", "Church events",
                          "college basketball games", "an egg hunt", "Lighting of the capitol christmas tree",
                          "Train rides", "theater events", "house decorating contests", "Car Shows",
                          "Hip - hop concert", "wine event", "a jewelry festival", "Boat fishing", "a pumpkin party",
                          "Holiday", "Open casting calls", "car seat checks", "egg hunts", "food and wine festival",
                          "wine dinners", "duck tour", "Music events", "Movie releases", "swimming activities",
                          "grand openings", "food truck friday", "a carnival", "Cooking baking parties",
                          "book signing events", "Fashion", "comic book events", "Farmers market", "program",
                          "stand - up performances", "dance parties", "lighting the Christmas tree", "hikes",
                          "horse shows", "Stand up comedy performances", "a country western band playing",
                          "boxing event", "the fair", "tree lighting events", "showing", "Food tasting",
                          "holiday activities", "food and music", "Block parties", "yoga related events",
                          "Fireworks shows", "Paint and sip parties", "performances", "Stand up Comedy", "Reunion",
                          "Pumpkin painting events", "organized snowball fights", "Dog shows", "Farmer 's Market",
                          "Gaming Tournaments", "Sleigh rides", "karaoke", "the trolls play production", "tree events",
                          "Gaming BBQs", "hay rides", "career fairs", "anime concerts", "stargazing", "rock bands",
                          "Movie premieres", "special sales", "cheese and wine festival", "Winter Gala",
                          "Cooking classes", "Food truck", "COncerts", "Comic Con", "the Collier County Fair",
                          "R & B concerts", "hockey events", "food truck festival", "ugly christmas sweater party",
                          "wedding conventions", "race", "live reggae bands", "Cooking events", "Gun show",
                          "Mother / daughter events", "the Marathon", "symphony concerts", "a tree lighting ceremony",
                          "the Winter Fest", "Holiday cocktail parties", "fireworks", "open Mic", "Festival of lights",
                          "business meetings", "Christmas made in the South", "House music events", "the play", "Expos",
                          "live bands playing", "theater", "Marathons", "Ornament making parties",
                          "Cosmetology classes", "Adventurous things", "Comedy Movies", "royal wedding",
                          "Polar Express events", "Craft show", "activties", "the grand opening",
                          "a magic the gathering meet up", "art show", "caroling", "light showings", "the dance party",
                          "orchestra performance", "Birthday party", "Karaoke night", "a sleigh ride", "play",
                          "the festival of trees", "movies playing", "artsy events", "soccer game gatherings",
                          "the lbgc parade", "tree lighting ceremony", "5K run", "gun shows", "races", "Music shows",
                          "Athletic events", "Classes", "Animal adoptions", "Gift giving parties", "punk bands",
                          "acid jazz", "classes", "Nightlife", "THE BEER FESTIVAL", "trick or treating",
                          "Plays showing", "Cookie decorating parties", "Job fairs", "indie music festivals",
                          "Frozen On Ice", "concert events", "dance classes", "song concerts", "zoo lights",
                          "book signings", "beach parties", "Food and Wine festivals", "opening",
                          "pumpkin carving events", "Boy Scout camping", "Dog Shows", "Make up tutorial events",
                          "the nutcracker play", "dance", "College Championship parties", "Tree lighting events",
                          "movie premieres", "rabies clinics", "carnivals", "winter wonderlands", "Fall Foliage Tours",
                          "the Boat Parade of lights", "Bulgarian festival", "Pizza Festival", "Gaming Meet and Swaps",
                          "help the homeless", "wizardcon", "Sports even", "Winter festivals", "Folk concerts",
                          "snowball dances", "famous punk bands", "Ice skating competitions", "Disney on ice",
                          "video game competitions", "Nutcracker concert", "the fireworks", "holiday concerts",
                          "skiing", "Fireworks", "The Terminator movie events", "Karaoke events", "Ice Skating Shows",
                          "pumpkin carving parties", "Symphonies", "Film Festival", "Puppy training classes",
                          "lady craft nights", "Landscape design conferences", "winter parties", "snow park",
                          "Stand up shows", "Ballgame viewing parties", "silent yoga", "cause walks", "rock concerts",
                          "Paint night", "The Nutcracker", "Chocolate Tours", "birthday parties", "hear music",
                          "birthday", "90's themed parties", "Holiday events", "flea market", "video game competition",
                          "Bethlehem tours", "strawberry festival", "anniversary events", "yoga master classes",
                          "Light shows", "Sip and paint parties", "anime events", "horse and buggy rides",
                          "E - Sports cookouts", "Tree Lightening Ceremonies", "Horse shows", "estate sales",
                          "a live band", "spring training game", "strolls", "Romantic events", "eclipse", "Bad Mom 2",
                          "live music", "Folk music concerts", "santa con", "Gallery openings",
                          "Wine and painting events", "author signings", "sport activities", "pancake breakfasts",
                          "a wine tasting", "Harry Potter symphony concert", "beer fest", "Singing competitions",
                          "Movies playing", "specials on tv", "match event", "flights HI - NY", "toddler classes",
                          "bands are playing", "Chocolate events", "present wrapping events", "Dancing club events",
                          "basketball games", "country stars", "Farmers Market", "live bands", "tree decorating party",
                          "the rodeo", "craft shows", "Recitals", "decorating parties", "art exhibitions",
                          "international food events", "tree light up event", "Wedding receptions", "Steampunk events",
                          "a trolley tour showing christmas lights", "fundraising events", "ice skating show",
                          "Country concerts", "a cake decorating class", "running events", "lightening", "wine events",
                          "Baking class", "comedians performing", "Jumanji", "Plant Nite events", "Music Concerts",
                          "top roof party", "any christmas tree lightings", "the movie showings", "the Boat parade",
                          "shows", "football", "art festival", "the Holiday Festival of Lights", "lighting events",
                          "snorkeling class", "Social events", "Learning to flip houses events", "Doll show",
                          "dog walking clubs", "the Circus", "Rap artists", "ethnic food festivals", "raves",
                          "any bands", "Painting classes", "Olympics", "Stand up comedy shows", "Photos",
                          "school band performance", "poetry events", "Paint and sip events", "hiking excursions",
                          "Live shows", "Community holiday parties", "the nutcracker playing", "Formal events",
                          "housewarming parties", "bunch", "Movie events", "meet new people",
                          "the sand castle competition", "Beach festivals", "Winery events", "Holiday pictures",
                          "salsa music", "lectures", "cultural festivals", "hiking tours", "classic car events",
                          "live jazz music", "maze", "meet up event", "wine festival", "a department party",
                          "Star Wars The Last Jedi", "Computer classes", "the community Christmas Tree decoration",
                          "holiday", "Craft Workshops", "Jazz artists", "the gun show", "Pop concerts",
                          "ice - hockey instruction", "showing of the frosty the snowman", "live holiday music",
                          "E3 convention", "Speed dating events", "nail decorating", "Haunted House",
                          "Scrapbooking activities", "craft sales", "Rock music events", "college bands",
                          "Indie bands playing", "Broadway play", "slot tournaments", "Mini Marathon after parties",
                          "services", "the County fair", "Monster Jam", "tree decorating contests",
                          "Mystery dinner theater", "airshows", "gatherings", "bike events",
                          "classical music performance", "a metal embossing event", "book fairs", "Punk concerts",
                          "activities", "Raindeer visit", "Oktoberfest festivals", "stand up comedy shows",
                          "star wars movie", "Improv comedy shows", "Ice skating", "Shows", "the ball drop",
                          "Viking events", "prom parties", "a line dancing class", "nativity scene", "Hamilton",
                          "the Coca - Cola 400", "plays showing", "Winter carnival", "the Holiday events", "chef",
                          "New movie playing", "rv shows", "craft beer events", "Standup Comedy", "Wine tasting party",
                          "music party play time", "Music concerts", "festivals", "Opera", "Standup Comedy events",
                          "sport match", "Taco Tuesday", "Strawberry Festivals", "Children 's Choir Performances"]

        return random.choice(category_event)


def __get_existed_provider_from_mimesis(mimesis_generator) -> list[BaseProvider]:
    providers = []
    exclude = list(BaseProvider().__dict__.keys())
    # Exclude locale explicitly because
    # it is not a provider.
    exclude.append("locale")

    keys = list(mimesis_generator.__dict__.keys())
    for attr in keys:
        if attr not in exclude:
            if attr.startswith("_"):  # 这种的需要实例化，是locale-dependent的class
                providers.append(mimesis_generator.__getattr__(attr[1:]))
            else:  # 这种的直接取值，是locale-independent的class
                providers.append(getattr(mimesis_generator, attr))
    return providers


def __get_existed_data_func_from_mimesis_provider(provider: BaseProvider) -> dict[str, callable]:
    exclude = dir(BaseDataProvider)
    exclude.append('Meta')

    funcs = {}
    dct = type(provider).__dict__
    for fn in dct:
        if fn not in exclude and not fn.startswith('_'):
            fn_name = fn.replace('_', ' ')
            funcs[fn_name] = getattr(provider, fn)
    return funcs


def _get_existed_generator_mimesis(language=Locale.EN) -> dict[str, callable]:
    local_generator = Generic(locale=language)  # 指定语言为英文

    providers = __get_existed_provider_from_mimesis(local_generator)
    funcs = {}
    for p in providers:
        funcs.update(__get_existed_data_func_from_mimesis_provider(p))

    return funcs


def _get_existed_generator_faker(language='EN') -> dict[str, callable]:
    fake = Faker(locale=language)
    fake.add_provider(Weather_Temperatur_Provider)
    fake.add_provider(Amount_Provider)
    fake.add_provider(Price_Provider)
    fake.add_provider(Ordinal_Provider)
    fake.add_provider(Location_Provider)
    fake.add_provider(Event_Attribute_Provider)
    fake.add_provider(Reminder_Provider)
    funcs = {fn.replace('_', ' '): fake.__getattr__(fn)
             for fn in dir(fake.factories[0]) if not fn.startswith('_')}  # 正常情况len(factories)就是==1的
    return funcs


def _get_existed_generators(language='EN') -> dict[str, callable]:
    if language == 'EN':
        lang_faker = 'EN'
        lang_mimesis = Locale.EN
    else:
        raise NotImplementedError

    funcs = {}
    funcs.update(_get_existed_generator_faker(lang_faker))
    funcs.update(_get_existed_generator_mimesis(lang_mimesis))

    global _CUSTOM_CONCEPTS_MAPPING
    for k, v in _CUSTOM_CONCEPTS_MAPPING.items():
        funcs[k] = funcs[v]

    return funcs
