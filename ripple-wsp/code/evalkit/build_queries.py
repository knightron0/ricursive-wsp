"""Builds queries.json: 96 hand-written queries, shuffled with a fixed seed, split 48/16/32."""
import json, os, random

Q = {
"knowledge": [
 "What is the capital of Australia?",
 "Why is the sky blue?",
 "Who wrote Pride and Prejudice?",
 "What causes the seasons on Earth?",
 "How many bones are in the adult human body?",
 "What is photosynthesis?",
 "What is the difference between weather and climate?",
 "Who was the first person to walk on the Moon?",
 "What is the largest ocean on Earth?",
 "Why do we have leap years?",
 "What does DNA stand for?",
 "What is the speed of light?",
 "Why do cats purr?",
 "What is inflation in economics?",
 "How do vaccines work?",
 "What language is spoken in Brazil?",
],
"howto": [
 "How do I boil an egg?",
 "How can I remove a coffee stain from a shirt?",
 "How do I change a flat bicycle tire?",
 "How do I make a paper airplane?",
 "What is the best way to memorize a speech?",
 "How do I tie a tie?",
 "How do I start a vegetable garden?",
 "How do I back up photos from my phone?",
 "How can I fall asleep faster?",
 "How do I make a good cup of tea?",
 "How do I write a cover letter?",
 "How do I unclog a kitchen sink?",
 "How do I cook rice on the stove?",
 "How can I improve my handwriting?",
],
"advice": [
 "I have a job interview tomorrow. Any tips?",
 "How should I deal with a noisy neighbor?",
 "What should I do if I feel unmotivated at work?",
 "How can I save more money each month?",
 "My friend forgot my birthday. Should I say something?",
 "How do I make friends in a new city?",
 "What should I consider before adopting a dog?",
 "How can I stop procrastinating?",
 "Should I learn guitar or piano first?",
 "How do I politely decline an invitation?",
 "What is a good way to start exercising?",
 "How can I become a better listener?",
 "What should I pack for a weekend camping trip?",
 "How do I ask my boss for a raise?",
],
"math": [
 "What is 17 times 6?",
 "What is 15% of 80?",
 "If a train travels 60 miles per hour for 2.5 hours, how far does it go?",
 "What is the square root of 144?",
 "How many minutes are in a day?",
 "What is 2 to the power of 10?",
 "A shirt costs $40 and is 25% off. What is the sale price?",
 "What is the average of 4, 8, and 15?",
 "How many sides does a hexagon have?",
 "What is 1000 minus 387?",
 "Is 91 a prime number?",
 "Convert 5 kilometers to miles.",
],
"writing": [
 "Write a one-sentence story about a lost umbrella.",
 "Write a short thank-you note to a teacher.",
 "Give me a slogan for a bakery.",
 "Describe the ocean in two sentences.",
 "Write a short birthday message for my grandmother.",
 "Suggest a name for a black cat.",
 "Write a two-line poem about autumn.",
 "Write a short product description for a reusable water bottle.",
 "Give me a title for a mystery novel set in Venice.",
 "Write a brief apology for being late to a meeting.",
 "Write a welcome message for a new coworker.",
 "Describe a sunrise to someone who has never seen one.",
 "Tell me a joke about computers.",
 "Write a short toast for a wedding.",
],
"coding": [
 "How do I reverse a string in Python?",
 "What is the difference between a list and a tuple in Python?",
 "What does HTML stand for?",
 "How do I write a for loop in JavaScript?",
 "What is a null pointer?",
 "Explain what a git commit is.",
 "How do I read a file in Python?",
 "What is recursion?",
 "What is the difference between == and === in JavaScript?",
 "What does SQL SELECT do?",
 "What is an API?",
 "How do I check if a number is even in Python?",
 "What is Big O notation?",
 "What is the difference between a stack and a queue?",
],
"opinion": [
 "Is it better to read books or listen to audiobooks?",
 "What is your favorite season and why?",
 "Are cats or dogs better pets?",
 "Is breakfast really the most important meal of the day?",
 "Should kids have homework?",
 "Is it worth learning a second language?",
 "What makes a movie great?",
 "Is remote work better than working in an office?",
 "Do you think people should travel more?",
 "Is pineapple on pizza acceptable?",
 "What is the most useful skill to learn?",
 "Is it better to be an early bird or a night owl?",
],
}

def main():
    cats = list(Q)
    assert sum(len(v) for v in Q.values()) == 96
    flat = [q for c in cats for q in Q[c]]
    assert len(set(flat)) == 96
    rng = random.Random(0)
    # stratified: shuffle within category, deal proportionally 3:1:2 so each split spans all categories
    splits = {"train": [], "val": [], "test": []}
    pool = []
    for c in cats:
        qs = Q[c][:]
        rng.shuffle(qs)
        pool.append(qs)
    order = ["train", "test", "train", "val", "train", "test"]  # 3:1:2 per 6 dealt
    i = 0
    while any(pool):
        for qs in pool:
            if qs:
                splits[order[i % 6]].append(qs.pop())
                i += 1
    for v in splits.values():
        rng.shuffle(v)
    assert [len(splits[k]) for k in ("train", "val", "test")] == [48, 16, 32]
    assert len(set(splits["train"]) | set(splits["val"]) | set(splits["test"])) == 96
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries.json"), "w") as f:
        json.dump(splits, f, indent=1, ensure_ascii=False)
    print({k: len(v) for k, v in splits.items()})

if __name__ == "__main__":
    main()
