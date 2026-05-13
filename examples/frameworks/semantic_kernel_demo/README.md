<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Semantic Kernel Example

**Complexity:** 🟢 Beginner

A minimal example using Semantic Kernel showcasing a multi-agent travel planning system where an Itinerary Agent creates a travel schedule, a Budget Agent ensures cost compliance, and a Summarizer Agent formats the final itinerary. **Please note that we only support OpenAI models currently**.

## Table of Contents

- [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Adding Long-Term Memory](#adding-long-term-memory)

## Key Features

- **Semantic Kernel Framework Integration:** Demonstrates NeMo Agent Toolkit support for Microsoft's Semantic Kernel framework alongside other frameworks like LangChain/LangGraph.
- **Multi-Agent Travel Planning:** Shows three specialized agents working together - an Itinerary Agent for schedule creation, a Budget Agent for cost management, and a Summarizer Agent for final formatting.
- **Cross-Agent Coordination:** Demonstrates how different agents can collaborate on a complex task, with each agent contributing its specialized capabilities to the overall workflow.
- **Long-Term Memory Integration:** Includes optional Mem0 platform integration for persistent memory, allowing agents to remember user preferences (like vegan dining or luxury hotel preferences) across sessions.
- **OpenAI Model Support:** Showcases NeMo Agent Toolkit compatibility with OpenAI models through the Semantic Kernel framework integration.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

### Install this Workflow

From the root directory of the NeMo Agent Toolkit library, run the following commands:

```bash
uv pip install -e examples/frameworks/semantic_kernel_demo
```

### Set Up API Keys

You need to set your OpenAI API key as an environment variable to access OpenAI AI services:

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
```

## Adding Long-Term Memory

 With NeMo Agent Toolkit, adding Long Term Memory (LTM) is as simple as adding a new section in the configuration file.

Once you add the LTM configuration, export your Mem0 API key, which is a prerequisite for using the LTM service. To create an API key, refer to the instructions in the [Mem0 Platform Guide](https://docs.mem0.ai/platform/quickstart).

Once you have your API key, export it as follows:

```bash
export MEM0_API_KEY=<YOUR_MEM0_API_KEY>
```

Then, you can run the workflow with the LTM configuration as follows:

```bash
nat run --config_file examples/frameworks/semantic_kernel_demo/configs/config.yml --input "Create a 3-day travel itinerary for Tokyo in April, suggest hotels  within a USD 2000 budget. I like staying at expensive hotels and am vegan"
```

**Expected Workflow Output**
The workflow produces a large amount of output, the end of the output should contain something similar to the following:

```console
Workflow Result:
['Below is your final 3-day Tokyo itinerary along with a cost breakdown and special notes based on your preferences for upscale accommodations and vegan dining options. This plan keeps your overall USD 2000 budget in mind while highlighting luxury experiences and convenience.\n\n──────────────────────────────\nItinerary Overview\n──────────────────────────────\n• Trip dates: April 15 – April 18, 2024 (3 nights)\n• Location: Tokyo, Japan\n• Focus: Upscale hotel experience and vegan-friendly dining/activities\n• Estimated Total Budget: USD 2000\n\n──────────────────────────────\nDay 1 – Arrival & Check-In\n──────────────────────────────\n• Arrive in Tokyo and transfer to your hotel.\n• Check in at the Luxury Penthouse (approx. USD 250 per night). \n   - 3-night cost: ~USD 750.\n• Spend the evening settling in and reviewing your itinerary.\n• Budget note: Approximately USD 1250 remains for transportation, meals (vegan options), and other expenses.\n\n──────────────────────────────\nDay 2 – Exploring Tokyo\n──────────────────────────────\n• Morning:\n   - Enjoy a leisurely breakfast at a nearby vegan-friendly café.\n   - Visit local attractions (e.g., upscale districts like Ginza or cultural areas such as Asakusa).\n• Afternoon:\n   - Explore boutique shopping, art galleries, or gardens.\n   - Alternatively, join a guided tour that includes stops at renowned cultural spots.\n• Evening:\n   - Dine at a well-reviewed vegan restaurant.\n   - Return to your hotel for a relaxing night.\n• Budget note: Allocate funds carefully for either private tours or special dining spots that cater to vegan diets.\n\n──────────────────────────────\nDay 3 – Final Day & Departure\n──────────────────────────────\n• Morning:\n   - Enjoy a hearty vegan breakfast.\n   - Visit any remaining attractions or enjoy some leisure time shopping.\n• Afternoon:\n   - Return to your hotel to check out.\n   - Ensure your remaining funds cover any last-minute transit for departure.\n• Evening:\n   - Depart for the airport, completing your upscale Tokyo experience.\n\n──────────────────────────────\nCost Breakdown\n──────────────────────────────\n• Hotel (Luxury Penthouse): USD 250 per night × 3 = ~USD 750\n• Remaining Budget:\n   - Transportation, meals (vegan options), and incidental expenses: ~USD 1250\n   - This allows flexibility for private tours, upscale experiences, and vegan dining experiences.\n• Overall Estimated Expenditure: Within USD 2000\n\n──────────────────────────────\nAdditional Notes\n──────────────────────────────\n• Your preference for expensive or upscale stays has been prioritized with the Luxury Penthouse option.\n• Vegan dining suggestions can be explored further by researching local vegan-friendly restaurants or booking a specialized food tour.\n• If you’d like more detailed recommendations on transit options, precise activity booking, or additional upscale experiences (e.g., fine dining, traditional cultural performances), please let me know!\n\nThis plan gives you a luxury Tokyo experience within your budget while accommodating your vegan lifestyle. Enjoy your trip!']
```

Please note that it is normal to see the LLM produce some errors on occasion as it handles complex structured tool calls. The workflow will automatically attempt to correct and retry the failed tool calls.

Assuming we've successfully added our preference for vegan restaurants in the last prompt to the agent, let us attempt to retrieve a more personalized itinerary with vegan dining options:

```bash
nat run --config_file examples/frameworks/semantic_kernel_demo/configs/config.yml --input "On a 1-day travel itinerary for Tokyo in April, suggest restaurants I would enjoy."
```

**Expected Workflow Output**
```console
Workflow Result:
['Here’s your final one-day Tokyo itinerary for April, with high-quality vegan-friendly dining recommendations that blend seamlessly with your sightseeing plans, along with a cost breakdown:\n\n─────────────────────────────  \nItinerary Overview\n\nMorning/Breakfast – Ain Soph. Journey  \n• Start your day with a creative vegan breakfast. Enjoy dishes like hearty vegan pancakes or fresh smoothie bowls in a cozy atmosphere – an ideal energizer before hitting the city.  \n• Location: Options available in vibrant neighborhoods like Shinjuku or Ginza.\n\nMidday/Lunch – T’s Restaurant  \n• Savor a bowl of vegan ramen and other Japanese-inspired dishes. This spot is conveniently located near major transit hubs and popular attractions like the Imperial Palace, making it a perfect lunch stop.  \n• Location: Near Tokyo Station and central attractions.\n\nAfternoon Snack – Seasonal Cafe near Cherry Blossoms  \n• While sightseeing, particularly near parks like Ueno or along the Meguro River, take a break at a local boutique cafe. Enjoy a refreshing herbal tea and a light plant-based treat, complemented by the beautiful bloom of cherry blossoms.  \n• Location: In the vicinity of your chosen park or river stroll.\n\nEvening/Dinner – AIN SOPH. Soar (or Similar Venue)  \n• Conclude your day with an elegant dining experience. Indulge in innovative vegan courses that creatively reimagine traditional flavors, in a serene setting ideal for unwinding after a busy day.  \n• Location: Commonly found in stylish districts like Shinjuku.\n\n─────────────────────────────  \nCost Breakdown (Estimates per Person)\n\n1. Breakfast at Ain Soph. Journey: ¥1,000–¥1,500  \n2. Lunch at T’s Restaurant: ¥800–¥1,300  \n3. Afternoon Snack at a Seasonal Cafe: ¥300–¥500  \n4. Dinner at AIN SOPH. Soar: ¥1,500–¥2,000  \n\nTotal Estimated Daily Dining Cost: Approximately ¥3,600–¥5,300 per person\n\n─────────────────────────────  \nAdditional Notes\n\n• Timing Tip: Plan your park visits for early morning or later afternoon to enjoy the cherry blossoms with fewer crowds and ideal light.  \n• Transportation: Utilize Tokyo’s efficient subway system to seamlessly move between Shinjuku, Ginza, Ueno, or other districts, ensuring you maximize your day.  \n• Reservations: It is advisable to reserve tables at popular spots like Ain Soph. Journey and AIN SOPH. Soar during the busy cherry blossom season.  \n• Dietary Focus: Each restaurant has been selected for its innovation with vegan-friendly menus, ensuring that each dining experience complements your travel itinerary.\n\n─────────────────────────────  \nEnjoy your one-day trip in Tokyo this April with delicious, thoughtfully curated dining stops and memorable sightseeing opportunities!']
```

The above output demonstrates that the agent was able to draw from memory to provide vegan-friendly recommendations.

Note: The long-term memory feature relies on LLM-based tool invocation, which can occasionally be non-deterministic. If you notice that the memory functionality isn't working as expected (e.g., the agent doesn't remember your preferences), try these solutions:
* Re-run your first and second inputs to ensure proper tool invocation
* Fine-tune the `long_term_memory_instructions` section in `config.yml` to better guide the agent's memory usage

These steps will help ensure your preferences are correctly stored and retrieved by the agent.
