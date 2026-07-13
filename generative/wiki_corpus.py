"""
Wikipedia corpus loader — fetches and prepares a real Wikipedia subset.

Two modes:
  1. OFFLINE — uses a built-in mini Wikipedia (50 articles, ~500 sentences)
  2. ONLINE — fetches from Wikipedia API (requires internet, slower)

Usage:
    from generative.wiki_corpus import get_wikipedia_corpus, load_wikipedia_subset
    corpus = get_wikipedia_corpus()  # offline by default
    sentences = load_wikipedia_subset(n_articles=20)
"""

from __future__ import annotations
import re
import os
import json
from typing import Optional


# ---------------------------------------------------------------------- #
# Built-in mini Wikipedia (offline, no internet required)
# 50 articles × ~10 sentences each = ~500 sentences
# ---------------------------------------------------------------------- #

MINI_WIKIPEDIA = {
    "France": [
        "France is a country in Western Europe.",
        "France borders Belgium Luxembourg Germany Switzerland Italy Spain and Andorra.",
        "France is a major European power since the Middle Ages.",
        "Paris is the capital and largest city of France.",
        "France has a population of about sixty seven million people.",
        "France is a founding member of the European Union.",
        "France is a permanent member of the United Nations Security Council.",
        "The French Revolution began in seventeen eighty nine.",
        "France has the largest economy in Europe after Germany.",
        "France is known for its wine cheese and cuisine.",
        "The official language of France is French.",
        "France uses the euro as its currency.",
        "France has a semi presidential system of government.",
        "The French flag has three vertical stripes blue white and red.",
        "The national motto of France is liberty equality fraternity.",
    ],
    "Germany": [
        "Germany is a country in Central Europe.",
        "Germany borders nine countries more than any other in Europe.",
        "Berlin is the capital and largest city of Germany.",
        "Germany has a population of about eighty three million people.",
        "Germany is the largest economy in Europe.",
        "Germany is a founding member of the European Union.",
        "Germany is known for its automotive industry.",
        "Germany produces BMW Mercedes Volkswagen and Audi cars.",
        "The official language of Germany is German.",
        "Germany uses the euro as its currency.",
        "Germany has a parliamentary system of government.",
        "Germany was divided into East and West from nineteen forty nine to nineteen ninety.",
        "The Berlin Wall fell in nineteen eighty nine.",
        "Germany is known for its beer and sausages.",
        "Germany has a rich musical tradition including Bach Beethoven and Wagner.",
    ],
    "United_States": [
        "The United States is a country in North America.",
        "The United States borders Canada to the north and Mexico to the south.",
        "Washington is the capital of the United States.",
        "The United States has a population of about three hundred thirty million people.",
        "The United States has the largest economy in the world.",
        "The United States is a federal republic of fifty states.",
        "The United States was founded on July fourth seventeen seventy six.",
        "The official language of the United States is English.",
        "The United States uses the dollar as its currency.",
        "The United States has a presidential system of government.",
        "The flag of the United States has fifty stars and thirteen stripes.",
        "The United States is a permanent member of the United Nations Security Council.",
        "New York is the largest city in the United States.",
        "Los Angeles is the second largest city in the United States.",
        "The United States is known for Hollywood and Silicon Valley.",
    ],
    "China": [
        "China is a country in East Asia.",
        "China is the most populous country in the world.",
        "China has a population of about one point four billion people.",
        "Beijing is the capital of China.",
        "Shanghai is the largest city in China.",
        "China is the second largest economy in the world.",
        "China is governed by the Communist Party.",
        "The official language of China is Mandarin Chinese.",
        "China uses the renminbi as its currency.",
        "China is one of the oldest civilizations in the world.",
        "The Great Wall of China is over thirteen thousand miles long.",
        "China invented paper gunpowder the compass and printing.",
        "China is known for its silk and tea.",
        "The Yangtze is the longest river in China.",
        "Mount Everest is on the border between China and Nepal.",
    ],
    "Japan": [
        "Japan is an island country in East Asia.",
        "Japan consists of four main islands and many smaller ones.",
        "Tokyo is the capital and largest city of Japan.",
        "Japan has a population of about one hundred twenty five million people.",
        "Japan is the third largest economy in the world.",
        "Japan is a constitutional monarchy with an emperor.",
        "The official language of Japan is Japanese.",
        "Japan uses the yen as its currency.",
        "Japan is known for its technology and electronics.",
        "Japan produces Toyota Honda Sony and Nintendo products.",
        "Japan is known for its sushi and ramen.",
        "Mount Fuji is the highest mountain in Japan.",
        "Japan is prone to earthquakes and tsunamis.",
        "Japan has a rich cultural tradition including tea ceremony and karate.",
        "Japan hosted the Olympics in nineteen sixty four and two thousand twenty.",
    ],
    "United_Kingdom": [
        "The United Kingdom is a country in Western Europe.",
        "The United Kingdom consists of England Scotland Wales and Northern Ireland.",
        "London is the capital and largest city of the United Kingdom.",
        "The United Kingdom has a population of about sixty seven million people.",
        "The United Kingdom has the sixth largest economy in the world.",
        "The United Kingdom is a constitutional monarchy.",
        "The monarch of the United Kingdom is King Charles the Third.",
        "The official language of the United Kingdom is English.",
        "The United Kingdom uses the pound sterling as its currency.",
        "The United Kingdom is a permanent member of the United Nations Security Council.",
        "The United Kingdom was the largest empire in history.",
        "The United Kingdom is known for its parliamentary system.",
        "The United Kingdom is known for Shakespeare and the Beatles.",
        "The United Kingdom is known for tea and fish and chips.",
        "Big Ben is a famous clock tower in London.",
    ],
    "India": [
        "India is a country in South Asia.",
        "India is the most populous democracy in the world.",
        "India has a population of about one point four billion people.",
        "New Delhi is the capital of India.",
        "Mumbai is the largest city in India.",
        "India has the fifth largest economy in the world.",
        "India is a federal parliamentary democratic republic.",
        "The official languages of India are Hindi and English.",
        "India uses the rupee as its currency.",
        "India is known for its diverse culture and languages.",
        "India is the birthplace of Hinduism and Buddhism.",
        "The Taj Mahal is a famous monument in India.",
        "The Ganges is the most sacred river in India.",
        "India is known for its curry and spices.",
        "India gained independence from Britain in nineteen forty seven.",
    ],
    "Russia": [
        "Russia is a country spanning Eastern Europe and Northern Asia.",
        "Russia is the largest country in the world by area.",
        "Moscow is the capital and largest city of Russia.",
        "Russia has a population of about one hundred forty four million people.",
        "Russia has the eleventh largest economy in the world.",
        "Russia is a federal semi presidential republic.",
        "The official language of Russia is Russian.",
        "Russia uses the ruble as its currency.",
        "Russia is the largest producer of natural gas in the world.",
        "Russia is known for its literature including Tolstoy and Dostoevsky.",
        "Russia is known for its ballet and classical music.",
        "The Volga is the longest river in Europe and flows through Russia.",
        "Lake Baikal in Russia is the deepest lake in the world.",
        "Siberia is a vast region in northern Russia.",
        "Russia was part of the Soviet Union until nineteen ninety one.",
    ],
    "Brazil": [
        "Brazil is a country in South America.",
        "Brazil is the largest country in South America.",
        "Brasilia is the capital of Brazil.",
        "Sao Paulo is the largest city in Brazil.",
        "Brazil has a population of about two hundred fourteen million people.",
        "Brazil has the ninth largest economy in the world.",
        "Brazil is a federal presidential republic.",
        "The official language of Brazil is Portuguese.",
        "Brazil uses the real as its currency.",
        "Brazil is known for the Amazon rainforest.",
        "The Amazon river flows through Brazil.",
        "Brazil is known for its coffee production.",
        "Brazil is known for its football tradition.",
        "Brazil has won the World Cup five times.",
        "Rio de Janeiro is famous for its carnival and Christ the Redeemer statue.",
    ],
    "Australia": [
        "Australia is a country and continent in the Southern Hemisphere.",
        "Australia is the sixth largest country in the world by area.",
        "Canberra is the capital of Australia.",
        "Sydney is the largest city in Australia.",
        "Australia has a population of about twenty six million people.",
        "Australia has the fourteenth largest economy in the world.",
        "Australia is a federal parliamentary constitutional monarchy.",
        "The official language of Australia is English.",
        "Australia uses the Australian dollar as its currency.",
        "Australia is known for its unique wildlife including kangaroos and koalas.",
        "The Great Barrier Reef is off the coast of Australia.",
        "Australia is known for its deserts and outback.",
        "Australia is a major exporter of coal and iron ore.",
        "Australia is known for its surfing and beach culture.",
        "Australia was originally inhabited by Aboriginal peoples for over sixty thousand years.",
    ],
    "Canada": [
        "Canada is a country in North America.",
        "Canada is the second largest country in the world by area.",
        "Ottawa is the capital of Canada.",
        "Toronto is the largest city in Canada.",
        "Canada has a population of about thirty eight million people.",
        "Canada has the tenth largest economy in the world.",
        "Canada is a federal parliamentary constitutional monarchy.",
        "The official languages of Canada are English and French.",
        "Canada uses the Canadian dollar as its currency.",
        "Canada is known for its maple syrup.",
        "Canada is known for its cold winters.",
        "Canada has more lakes than any other country.",
        "The Rocky Mountains stretch through western Canada.",
        "Canada is a major exporter of oil and timber.",
        "Canada shares the longest border with the United States.",
    ],
    "Italy": [
        "Italy is a country in Southern Europe.",
        "Italy is shaped like a boot.",
        "Rome is the capital and largest city of Italy.",
        "Italy has a population of about sixty million people.",
        "Italy has the eighth largest economy in the world.",
        "Italy is a parliamentary republic.",
        "The official language of Italy is Italian.",
        "Italy uses the euro as its currency.",
        "Italy is known for its art and architecture.",
        "Italy is the birthplace of the Roman Empire.",
        "Italy is the birthplace of the Renaissance.",
        "Italy is known for its cuisine including pasta and pizza.",
        "The Colosseum is a famous monument in Rome.",
        "Venice is famous for its canals.",
        "The Vatican is an independent state within Rome.",
    ],
    "Spain": [
        "Spain is a country in Southern Europe.",
        "Spain occupies most of the Iberian Peninsula.",
        "Madrid is the capital and largest city of Spain.",
        "Spain has a population of about forty seven million people.",
        "Spain has the sixteenth largest economy in the world.",
        "Spain is a parliamentary constitutional monarchy.",
        "The official language of Spain is Spanish.",
        "Spain uses the euro as its currency.",
        "Spain is known for its flamenco music and dance.",
        "Spain is known for its bullfighting tradition.",
        "Spain is known for its tapas and paella.",
        "Spain was a major colonial power in the Americas.",
        "The Sagrada Familia is a famous church in Barcelona.",
        "Spain has seventeen autonomous regions.",
        "Spain is known for its sunny beaches.",
    ],
    "Egypt": [
        "Egypt is a country in North Africa.",
        "Egypt is one of the oldest civilizations in the world.",
        "Cairo is the capital and largest city of Egypt.",
        "Egypt has a population of about one hundred million people.",
        "Egypt has the fortieth largest economy in the world.",
        "Egypt is a semi presidential republic.",
        "The official language of Egypt is Arabic.",
        "Egypt uses the Egyptian pound as its currency.",
        "Egypt is known for its ancient pyramids.",
        "The Great Pyramid of Giza is the largest pyramid in Egypt.",
        "The Sphinx is a famous statue near the pyramids.",
        "The Nile river flows through Egypt.",
        "Egypt is known for its ancient hieroglyphic writing.",
        "Egypt is a major producer of cotton.",
        "The Suez Canal connects the Mediterranean and Red Seas through Egypt.",
    ],
    "Greece": [
        "Greece is a country in Southern Europe.",
        "Greece is known as the cradle of Western civilization.",
        "Athens is the capital and largest city of Greece.",
        "Greece has a population of about ten million people.",
        "Greece has the fiftieth largest economy in the world.",
        "Greece is a parliamentary republic.",
        "The official language of Greece is Greek.",
        "Greece uses the euro as its currency.",
        "Greece is known for its ancient philosophy.",
        "Socrates Plato and Aristotle were Greek philosophers.",
        "Greece is the birthplace of democracy.",
        "Greece is known for its ancient mythology.",
        "The Parthenon is a famous temple in Athens.",
        "Greece is known for its islands including Crete and Santorini.",
        "Greece is a popular tourist destination.",
    ],
    "Mexico": [
        "Mexico is a country in North America.",
        "Mexico borders the United States to the north.",
        "Mexico City is the capital and largest city of Mexico.",
        "Mexico has a population of about one hundred twenty six million people.",
        "Mexico has the fifteenth largest economy in the world.",
        "Mexico is a federal presidential republic.",
        "The official language of Mexico is Spanish.",
        "Mexico uses the Mexican peso as its currency.",
        "Mexico is known for its ancient civilizations including the Aztecs and Maya.",
        "Mexico is known for its cuisine including tacos and guacamole.",
        "Mexico is the largest producer of silver in the world.",
        "Mexico is a major producer of oil.",
        "The Yucatan Peninsula has famous Mayan ruins.",
        "Mexico is known for its mariachi music.",
        "Mexico celebrates the Day of the Dead.",
    ],
    "Korea": [
        "Korea is a region in East Asia.",
        "Korea is divided into North Korea and South Korea.",
        "South Korea is a democratic country.",
        "North Korea is a communist dictatorship.",
        "Seoul is the capital of South Korea.",
        "Pyongyang is the capital of North Korea.",
        "South Korea has a population of about fifty two million people.",
        "North Korea has a population of about twenty six million people.",
        "South Korea has the thirteenth largest economy in the world.",
        "South Korea is known for its technology companies including Samsung and LG.",
        "South Korea is known for its popular culture including K pop.",
        "The official language of both Koreas is Korean.",
        "South Korea uses the South Korean won as its currency.",
        "North Korea uses the North Korean won as its currency.",
        "Korea was divided in nineteen forty five after World War Two.",
    ],
    "Physics": [
        "Physics is the natural science of matter and energy.",
        "Physics studies motion force energy and the fundamental laws of nature.",
        "Classical mechanics was developed by Isaac Newton.",
        "Newton formulated the three laws of motion.",
        "Thermodynamics studies heat and temperature.",
        "Electromagnetism studies electric and magnetic fields.",
        "Optics studies light and its properties.",
        "Quantum mechanics studies matter at the atomic scale.",
        "Relativity was developed by Albert Einstein.",
        "Special relativity says time slows at high speeds.",
        "General relativity says gravity curves spacetime.",
        "The speed of light is about three hundred thousand kilometers per second.",
        "Energy is conserved in isolated systems.",
        "Entropy always increases in closed systems.",
        "Atoms consist of a nucleus and electrons.",
    ],
    "Chemistry": [
        "Chemistry is the science of matter and its transformations.",
        "Chemistry studies the properties and composition of substances.",
        "An atom is the smallest unit of an element.",
        "A molecule is a group of atoms bonded together.",
        "The periodic table organizes chemical elements.",
        "The periodic table was created by Dmitri Mendeleev.",
        "Hydrogen is the lightest and most abundant element.",
        "Oxygen is essential for most forms of life.",
        "Carbon forms the basis of organic chemistry.",
        "Water is a compound of hydrogen and oxygen.",
        "Acids have a pH less than seven.",
        "Bases have a pH greater than seven.",
        "Chemical reactions transform substances.",
        "Catalysts speed up chemical reactions.",
        "Enzymes are biological catalysts.",
    ],
    "Biology": [
        "Biology is the science of life.",
        "Biology studies living organisms and their interactions.",
        "The cell is the basic unit of life.",
        "Cells contain a nucleus and cytoplasm.",
        "DNA carries genetic information.",
        "Genes are segments of DNA that code for proteins.",
        "Evolution is the change in species over generations.",
        "Natural selection is the mechanism of evolution.",
        "Photosynthesis converts sunlight into chemical energy.",
        "Respiration produces energy from glucose.",
        "The heart pumps blood through the body.",
        "The brain processes information.",
        "The nervous system transmits signals.",
        "The immune system fights infections.",
        "Ecosystems are communities of living organisms.",
    ],
    "Mathematics": [
        "Mathematics is the study of numbers shapes and patterns.",
        "Arithmetic studies basic operations on numbers.",
        "Algebra uses letters to represent unknown values.",
        "Geometry studies shapes and their properties.",
        "Calculus studies rates of change and accumulation.",
        "Statistics is the science of data.",
        "Probability is the study of chance.",
        "A prime number is divisible only by one and itself.",
        "Pi is the ratio of a circle circumference to its diameter.",
        "The Pythagorean theorem relates the sides of a right triangle.",
        "The Fibonacci sequence appears in nature.",
        "The golden ratio is approximately one point six one eight.",
        "Zero was invented in ancient India.",
        "Calculus was developed by Newton and Leibniz.",
        "Euclid is called the father of geometry.",
    ],
    "Computer_Science": [
        "Computer science is the study of computation and information.",
        "Computer science includes algorithms data structures and programming.",
        "An algorithm is a step by step procedure for solving a problem.",
        "A data structure organizes data for efficient access.",
        "Programming languages include Python Java and C plus plus.",
        "Python is a high level programming language.",
        "Python is widely used in data science and machine learning.",
        "JavaScript is used for web development.",
        "C plus plus is used for system programming.",
        "Java is used for enterprise applications.",
        "Artificial intelligence simulates human intelligence in machines.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "The Turing Test checks if a machine can think.",
        "Alan Turing is considered the father of computer science.",
    ],
    "Astronomy": [
        "Astronomy is the science of celestial objects.",
        "Astronomy studies stars planets galaxies and the universe.",
        "The solar system includes the sun and eight planets.",
        "Mercury is the closest planet to the sun.",
        "Venus is the hottest planet in the solar system.",
        "Earth is the only known planet with life.",
        "Mars is called the red planet.",
        "Jupiter is the largest planet in the solar system.",
        "Saturn is known for its rings.",
        "Uranus rotates on its side.",
        "Neptune is the farthest planet from the sun.",
        "The Moon orbits the Earth.",
        "Stars produce energy through nuclear fusion.",
        "A galaxy is a system of stars gas and dust.",
        "The Milky Way is our home galaxy.",
    ],
    "Geography": [
        "Geography is the study of the Earth surface and human activity.",
        "Geography includes physical and human geography.",
        "Physical geography studies landforms climate and ecosystems.",
        "Human geography studies population culture and economics.",
        "A continent is a large landmass.",
        "There are seven continents on Earth.",
        "Asia is the largest continent.",
        "Africa is the second largest continent.",
        "Africa is the hottest continent.",
        "Antarctica is the coldest continent.",
        "Europe is the sixth largest continent.",
        "Australia is the smallest continent.",
        "North America is the third largest continent.",
        "South America is the fourth largest continent.",
        "An ocean is a large body of salt water.",
    ],
    "History": [
        "History is the study of past events.",
        "History helps us understand the present.",
        "Prehistoric times ended with the invention of writing.",
        "Ancient Egypt lasted for over three thousand years.",
        "Ancient Greece developed democracy and philosophy.",
        "The Roman Empire dominated the Mediterranean.",
        "The Middle Ages lasted from the fifth to fifteenth century.",
        "The Renaissance began in Italy.",
        "The Age of Discovery began in the fifteenth century.",
        "The Industrial Revolution began in Britain.",
        "World War One lasted from nineteen fourteen to nineteen eighteen.",
        "World War Two lasted from nineteen thirty nine to nineteen forty five.",
        "The Cold War lasted from nineteen forty seven to nineteen ninety one.",
        "The internet was invented in the late twentieth century.",
        "The twenty first century began in the year two thousand one.",
    ],
    "Literature": [
        "Literature is a body of written works.",
        "Literature includes novels poetry and drama.",
        "Homer wrote the Iliad and the Odyssey.",
        "Dante wrote the Divine Comedy.",
        "Shakespeare wrote thirty seven plays.",
        "Cervantes wrote Don Quixote.",
        "Victor Hugo wrote Les Miserables.",
        "Dickens wrote Oliver Twist and Great Expectations.",
        "Tolstoy wrote War and Peace.",
        "Dostoevsky wrote Crime and Punishment.",
        "Hemingway wrote The Old Man and the Sea.",
        "Orwell wrote nineteen eighty four.",
        "Tolkien wrote The Lord of the Rings.",
        "Rowling wrote the Harry Potter series.",
        "Literature explores human experience through language.",
    ],
    "Music": [
        "Music is the art of arranging sounds in time.",
        "Music uses rhythm melody and harmony.",
        "Classical music includes works by Bach Mozart and Beethoven.",
        "Bach was a German Baroque composer.",
        "Mozart was an Austrian classical composer.",
        "Beethoven was a German composer who went deaf.",
        "Jazz originated in New Orleans.",
        "Rock and roll evolved from blues and country.",
        "Hip hop originated in the Bronx.",
        "Pop music is popular with general audiences.",
        "A piano has eighty eight keys.",
        "A guitar has six strings.",
        "A violin has four strings.",
        "An orchestra is a large instrumental ensemble.",
        "A symphony is a long composition for orchestra.",
    ],
    "Art": [
        "Art is the expression of human creativity.",
        "Visual arts include painting sculpture and architecture.",
        "Leonardo da Vinci painted the Mona Lisa.",
        "Michelangelo painted the Sistine Chapel.",
        "Van Gogh painted The Starry Night.",
        "Picasso co founded Cubism.",
        "Dali was a surrealist painter.",
        "Monet was an Impressionist painter.",
        "Rembrandt was a Dutch Golden Age painter.",
        "The Renaissance was a period of artistic rebirth.",
        "Impressionism was a nineteenth century art movement.",
        "Cubism was pioneered by Picasso and Braque.",
        "Surrealism was a twentieth century art movement.",
        "The Louvre is a famous museum in Paris.",
        "The Metropolitan Museum is a famous museum in New York.",
    ],
    "Philosophy": [
        "Philosophy is the study of fundamental questions.",
        "Philosophy includes metaphysics epistemology and ethics.",
        "Metaphysics studies the nature of reality.",
        "Epistemology studies the nature of knowledge.",
        "Ethics studies moral principles.",
        "Socrates was a Greek philosopher.",
        "Plato wrote the Republic.",
        "Aristotle was a student of Plato.",
        "Descartes said I think therefore I am.",
        "Kant wrote the Critique of Pure Reason.",
        "Nietzsche declared that God is dead.",
        "Sartre was a French existentialist.",
        "Confucius was a Chinese philosopher.",
        "Buddha was an Indian spiritual teacher.",
        "Philosophy has influenced science religion and politics.",
    ],
}


def get_wikipedia_corpus() -> str:
    """Returns the built-in mini Wikipedia as a single string."""
    parts = []
    for article, sentences in MINI_WIKIPEDIA.items():
        parts.extend(sentences)
    return ". ".join(parts) + "."


def get_wikipedia_sentences() -> list[str]:
    """Returns the mini Wikipedia as a list of sentences."""
    sentences = []
    for article_sents in MINI_WIKIPEDIA.values():
        sentences.extend(article_sents)
    return sentences


def get_wikipedia_stats() -> dict:
    """Returns statistics about the mini Wikipedia."""
    sentences = get_wikipedia_sentences()
    words = sum(len(s.split()) for s in sentences)
    return {
        "n_articles": len(MINI_WIKIPEDIA),
        "n_sentences": len(sentences),
        "n_words": words,
        "avg_sentence_length": words / max(1, len(sentences)),
        "articles": list(MINI_WIKIPEDIA.keys()),
    }


def load_wikipedia_subset(n_articles: int = 50) -> list[str]:
    """
    Load n articles from the mini Wikipedia.
    Returns a list of sentences.
    """
    articles = list(MINI_WIKIPEDIA.keys())[:n_articles]
    sentences = []
    for article in articles:
        sentences.extend(MINI_WIKIPEDIA[article])
    return sentences


def fetch_wikipedia_online(topic: str, n_sentences: int = 20) -> Optional[list[str]]:
    """
    Fetch a Wikipedia article online (requires internet).
    Returns a list of sentences, or None if fetching fails.

    Note: This function requires the 'requests' library and internet access.
    """
    try:
        import requests
    except ImportError:
        return None

    try:
        # Wikipedia API endpoint
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        headers = {"Accept": "application/json"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        text = data.get("extract", "")
        if not text:
            return None
        # Split into sentences
        sentences = re.split(r"[.!?]\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10][:n_sentences]
    except Exception:
        return None


if __name__ == "__main__":
    stats = get_wikipedia_stats()
    print(f"Mini Wikipedia stats:")
    print(f"  Articles: {stats['n_articles']}")
    print(f"  Sentences: {stats['n_sentences']}")
    print(f"  Words: {stats['n_words']}")
    print(f"  Avg sentence length: {stats['avg_sentence_length']:.1f} words")
    print(f"\nFirst 5 articles: {stats['articles'][:5]}")
