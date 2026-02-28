from crewai import Agent , Task , Crew
from crewai.project import CrewBase , agent , crew , task
from crewai.tools import BaseTool
from serpapi import GoogleSearch   

from dotenv import load_dotenv
load_dotenv()
import os

import sys
print("Python path:", sys.executable)

import pydantic
print("Pydantic version:", pydantic.__version__)

class SerpAPITool(BaseTool):
    name: str = "Search Internet with SerpAPI"
    description: str = "Search the internet for latest information"

    def _run(self, query: str):
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": 5
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        output = ""

        if "organic_results" in results:
            for result in results["organic_results"]:
                title = result.get("title", "")
                link = result.get("link", "")
                snippet = result.get("snippet", "")
                output += f"\nTitle: {title}\nLink: {link}\nSnippet: {snippet}\n"

        return output

@CrewBase
class BlogCrew():
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def research_agent(self):
        return Agent(
           config = self.agents_config['research_agent'],
            tools = [SerpAPITool()],
            verbose = True,
        )
    
    @agent
    def writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['writing_agent'], # type: ignore[index]
            verbose=True
        )
    
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
            agent = self.research_agent()
        )

    @task
    def writer_task(self) -> Task:
        return Task(
            config=self.tasks_config['writer_task'], # type: ignore[index]
            agent = self.writer_agent()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.research_agent(), self.writer_agent()],
            tasks=[self.research_task(), self.writer_task()]
        )
    

if __name__ == "__main__":
    blog_crew = BlogCrew()
    blog_crew.crew().kickoff(inputs={"topic": "Future of AI(Artificial Intelligence)"})